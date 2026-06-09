from __future__ import annotations

import argparse
from pathlib import Path

from cutload_pipeline import ExperimentConfig, ExperimentRunner
from cutload_pipeline.config import MappingConfig, PathConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the refactored CNC cutting-load pipeline.")
    parser.add_argument("--sim-glob", default="data/synthetic/small/Sim_20mm/*.csv")
    parser.add_argument("--act-glob", default="data/synthetic/small/Act_20mm/*.csv")
    parser.add_argument("--output-dir", default="results/after")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        paths=PathConfig(sim_glob=args.sim_glob, act_glob=args.act_glob, output_dir=Path(args.output_dir)),
        mapping=MappingConfig(),
        train=TrainConfig(epochs=args.epochs, hidden_size=args.hidden_size, seed=args.seed),
    )
    result = ExperimentRunner(cfg).run()
    print(result)


if __name__ == "__main__":
    main()
