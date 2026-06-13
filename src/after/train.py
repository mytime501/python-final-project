from __future__ import annotations

import argparse
from pathlib import Path

from cutload_pipeline import ExperimentConfig, ExperimentRunner
from cutload_pipeline.config import MappingConfig, PathConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNC 절삭 부하 예측 파이프라인 실행")
    parser.add_argument("--sim-glob", required=True, metavar="폴더 경로", help="SIM CSV 파일 폴더 경로 또는 검색 패턴")
    parser.add_argument("--act-glob", required=True, metavar="폴더 경로", help="ACT CSV 파일 폴더 경로 또는 검색 패턴")
    parser.add_argument("--output-dir", default="결과_폴더", metavar="폴더 경로", help="결과를 저장할 폴더 경로")
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
