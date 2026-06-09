from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(slots=True)
class SimData:
    block: np.ndarray
    xyz: np.ndarray
    feats: np.ndarray
    s: np.ndarray
    block_indices: dict[int, np.ndarray]
    train_start_s: float
    train_end_s: float


@dataclass(slots=True)
class RunData:
    name: str
    mapped_raw: np.ndarray
    look_raw: np.ndarray
    y_raw: np.ndarray
    train_mask: np.ndarray
    t_ms: np.ndarray
    mapped: Optional[np.ndarray] = None
    look: Optional[np.ndarray] = None
    mapped_t: Optional[torch.Tensor] = None
    look_t: Optional[torch.Tensor] = None
    y_t: Optional[torch.Tensor] = None
    mask_t: Optional[torch.Tensor] = None

    def to_torch(self, device: torch.device) -> None:
        if self.mapped is None or self.look is None:
            raise ValueError("Scaler must be applied before tensor conversion.")
        self.mapped_t = torch.tensor(self.mapped, dtype=torch.float32, device=device)
        self.look_t = torch.tensor(self.look, dtype=torch.float32, device=device)
        self.y_t = torch.tensor(self.y_raw, dtype=torch.float32, device=device)
        self.mask_t = torch.tensor(self.train_mask, dtype=torch.bool, device=device)
