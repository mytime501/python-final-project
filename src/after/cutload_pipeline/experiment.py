from __future__ import annotations

import logging
from pathlib import Path

from .act_mapper import ActMapper
from .config import ExperimentConfig
from .decorators import log_call, memory_traced, timed
from .evaluator import Evaluator
from .exporter import Exporter
from .path_resolver import PathResolver
from .scaler import FeatureScaler
from .sim_loader import SimLoader
from .trainer import Trainer


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    @timed
    @memory_traced
    @log_call
    def run(self) -> dict[str, object]:
        resolver = PathResolver(self.cfg.paths.sim_glob, self.cfg.paths.act_glob)
        pairs = list(resolver.iter_pairs())
        if not pairs:
            raise RuntimeError("CSV 파일 쌍을 찾지 못했습니다. SIM 폴더 경로와 ACT 폴더 경로를 확인하세요.")
        loader = SimLoader()
        mapper = ActMapper(self.cfg.mapping)
        runs = [
            mapper.build_run(pair.name, loader.load(pair.sim_path), pair.act_path, min_rows=self.cfg.min_rows)
            for pair in pairs
        ]
        n_val = 1 if len(runs) >= 3 and self.cfg.val_ratio > 0 else 0
        train_runs = runs[n_val:] if n_val else runs
        val_runs = runs[:n_val]
        scaler = FeatureScaler.fit(train_runs)
        for run in runs:
            scaler.apply(run)
        trainer = Trainer(self.cfg.train, scaler)
        model = trainer.train(train_runs, val_runs=val_runs)
        eval_runs = val_runs or train_runs
        metrics = Evaluator(self.cfg.train, scaler).evaluate(model, eval_runs)
        exporter = Exporter(Path(self.cfg.paths.output_dir))
        exporter.save_scaler(scaler)
        exporter.save_metrics(metrics)
        return {"pairs": len(pairs), "train_runs": len(train_runs), "val_runs": len(val_runs), "metrics": metrics}
