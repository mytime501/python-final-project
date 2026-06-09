from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_models import RunData


@dataclass(slots=True)
class FeatureScaler:
    mapped_mean: np.ndarray
    mapped_std: np.ndarray
    look_mean: np.ndarray
    look_std: np.ndarray
    dy_mean: float
    dy_std: float

    @classmethod
    def fit(cls, runs: list[RunData], eps: float = 1e-8) -> "FeatureScaler":
        mapped = np.concatenate([r.mapped_raw for r in runs], axis=0)
        look = np.concatenate([r.look_raw for r in runs], axis=0)
        dy = np.concatenate([np.diff(r.y_raw, axis=0) for r in runs if len(r.y_raw) > 1], axis=0)
        return cls(
            mapped_mean=mapped.mean(axis=0),
            mapped_std=np.maximum(mapped.std(axis=0), eps),
            look_mean=look.mean(axis=0),
            look_std=np.maximum(look.std(axis=0), eps),
            dy_mean=0.0,
            dy_std=float(max(float(dy.std()) if dy.size else 1.0, eps)),
        )

    def apply(self, run: RunData) -> None:
        run.mapped = ((run.mapped_raw - self.mapped_mean) / self.mapped_std).astype(np.float32)
        run.look = ((run.look_raw - self.look_mean) / self.look_std).astype(np.float32)
