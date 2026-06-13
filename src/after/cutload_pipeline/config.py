from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathConfig:
    sim_glob: str = ""
    act_glob: str = ""
    output_dir: Path = Path("결과_폴더")


@dataclass(slots=True)
class MappingConfig:
    window: int = 300
    wz: float = 3.0
    allow_missing_block: bool = True
    lookahead_steps: int = 10


@dataclass(slots=True)
class TrainConfig:
    hidden_size: int = 32
    num_layers: int = 1
    lr: float = 1e-3
    epochs: int = 3
    tbptt_steps: int = 80
    device: str = "cpu"
    seed: int = 42


@dataclass(slots=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val_ratio: float = 0.25
    min_rows: int = 10
