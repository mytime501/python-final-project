from __future__ import annotations

import logging
import random
from typing import Iterator

import numpy as np
import torch

from .config import TrainConfig
from .data_models import RunData
from .model import GRUStepModel
from .scaler import FeatureScaler

LOGGER = logging.getLogger("cutload_pipeline")


def iter_train_steps(run: RunData) -> Iterator[int]:
    """Yield valid transition indices lazily for one stateful run."""
    assert run.y_t is not None and run.mask_t is not None
    for t in range(0, len(run.y_t) - 1):
        if bool(run.mask_t[t].item() and run.mask_t[t + 1].item()):
            yield t


class Trainer:
    def __init__(self, cfg: TrainConfig, scaler: FeatureScaler) -> None:
        self.cfg = cfg
        self.scaler = scaler

    def train(self, runs: list[RunData], val_runs: list[RunData] | None = None, target_runs: list[RunData] | None = None) -> GRUStepModel:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        device = torch.device(self.cfg.device)
        for run in runs + (val_runs or []) + (target_runs or []):
            run.to_torch(device)
        input_size = runs[0].mapped.shape[1] + runs[0].look.shape[1] + 1  # type: ignore[union-attr]
        model = GRUStepModel(input_size, self.cfg.hidden_size, self.cfg.num_layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        self._run_stage("train", model, opt, runs, max_epochs=self.cfg.epochs)
        if target_runs:
            self._run_stage("target_finetune", model, opt, runs, max_epochs=1, stage_runs=target_runs)
        return model

    def _run_stage(
        self,
        stage: str,
        model: GRUStepModel,
        optimizer: torch.optim.Optimizer,
        runs: list[RunData],
        max_epochs: int,
        stage_runs: list[RunData] | None = None,
    ) -> None:
        active_runs = stage_runs if stage_runs is not None else runs
        LOGGER.info("%s uses %d run(s)", stage, len(active_runs))
        model.train()
        device = torch.device(self.cfg.device)
        for _ in range(max_epochs):
            for run in active_runs:
                assert run.mapped_t is not None and run.look_t is not None and run.y_t is not None and run.mask_t is not None
                h = torch.zeros(self.cfg.num_layers, 1, self.cfg.hidden_size, device=device)
                resid = torch.zeros((1, 1), dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                losses: list[torch.Tensor] = []
                for t in iter_train_steps(run):
                    x_t = torch.cat([run.mapped_t[t : t + 1], run.look_t[t : t + 1], resid], dim=1)
                    dy_hat, h = model.forward_step(x_t, h)
                    dy_true = (run.y_t[t + 1 : t + 2] - run.y_t[t : t + 1]) / self.scaler.dy_std
                    losses.append(torch.nn.functional.mse_loss(dy_hat, dy_true))
                    resid = (dy_true - dy_hat).detach()
                    if len(losses) >= self.cfg.tbptt_steps:
                        loss = torch.stack(losses).mean()
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        h = h.detach()
                        losses.clear()
                if losses:
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    optimizer.step()
