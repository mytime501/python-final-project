from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scaler import FeatureScaler


class Exporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_scaler(self, scaler: FeatureScaler) -> Path:
        path = self.output_dir / "scaler_params.json"
        payload = {
            "mapped_mean": scaler.mapped_mean.tolist(),
            "mapped_std": scaler.mapped_std.tolist(),
            "look_mean": scaler.look_mean.tolist(),
            "look_std": scaler.look_std.tolist(),
            "dy_mean": scaler.dy_mean,
            "dy_std": scaler.dy_std,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_metrics(self, metrics: dict[str, float]) -> Path:
        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_numpy_predictions(self, y_pred: np.ndarray) -> Path:
        path = self.output_dir / "predictions.npy"
        np.save(path, y_pred)
        return path
