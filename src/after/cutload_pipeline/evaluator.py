from __future__ import annotations

import time

import numpy as np
import torch

from .config import TrainConfig
from .data_models import RunData
from .metrics import regression_metrics
from .model import GRUStepModel
from .scaler import FeatureScaler


class Evaluator:
    def __init__(self, cfg: TrainConfig, scaler: FeatureScaler) -> None:
        self.cfg = cfg
        self.scaler = scaler

    def evaluate(self, model: GRUStepModel, runs: list[RunData]) -> dict[str, float]:
        device = torch.device(self.cfg.device)
        model.eval()
        y_true: list[float] = []
        y_pred: list[float] = []
        times: list[float] = []
        with torch.no_grad():
            for run in runs:
                run.to_torch(device)
                assert run.mapped_t is not None and run.look_t is not None and run.y_t is not None
                h = torch.zeros(self.cfg.num_layers, 1, self.cfg.hidden_size, device=device)
                resid = torch.zeros((1, 1), dtype=torch.float32, device=device)
                for t in range(0, len(run.y_t) - 1):
                    start = time.perf_counter()
                    x_t = torch.cat([run.mapped_t[t : t + 1], run.look_t[t : t + 1], resid], dim=1)
                    dy_hat, h = model.forward_step(x_t, h)
                    times.append(time.perf_counter() - start)
                    pred = float(run.y_t[t].item() + dy_hat.item() * self.scaler.dy_std)
                    true = float(run.y_t[t + 1].item())
                    y_pred.append(pred)
                    y_true.append(true)
                    resid = torch.tensor([[(true - pred) / self.scaler.dy_std]], dtype=torch.float32, device=device)
        out = regression_metrics(np.array(y_true), np.array(y_pred))
        out["Avg pred. time (ms)"] = float(np.mean(times) * 1000.0) if times else 0.0
        return out
