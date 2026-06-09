from __future__ import annotations

import torch
from torch import nn


class GRUStepModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward_step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, h2 = self.gru(x_t.unsqueeze(1), h)
        return self.head(out[:, -1, :]), h2
