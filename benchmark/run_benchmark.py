from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import platform
import random
import statistics
import sys
import time
import tracemalloc
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "after"))

from cutload_pipeline.act_mapper import ActMapper
from cutload_pipeline.config import ExperimentConfig, MappingConfig, PathConfig, TrainConfig
from cutload_pipeline.experiment import ExperimentRunner
from cutload_pipeline.path_resolver import PathResolver
from cutload_pipeline.sim_loader import SimLoader


SIZE_SPECS = {
    "small": {"rows": 120, "pairs": 3},
    "medium": {"rows": 360, "pairs": 3},
    "large": {"rows": 720, "pairs": 3},
}
SIZES = list(SIZE_SPECS)


def _write_csv_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_synthetic(size: str, root: Path) -> Path:
    spec = SIZE_SPECS[size]
    out_dir = root / size
    sim_dir = out_dir / "Sim_20mm"
    act_dir = out_dir / "Act_20mm"
    rng = random.Random(20260609)
    for pair_idx in range(int(spec["pairs"])):
        feed = 900 + pair_idx * 150
        depth = 0.5 + pair_idx * 0.25
        width = 2.0 + pair_idx * 0.2
        sim_rows: list[list[object]] = []
        act_rows: list[list[object]] = []
        rows = int(spec["rows"])
        for i in range(rows):
            block = i // 12 + 1
            theta = i / max(rows - 1, 1) * math.pi * 4
            x = i * 0.08
            y = math.sin(theta) * 0.8 + pair_idx
            z = -depth + math.cos(theta) * 0.03
            cutting = 8 <= i < rows - 8
            mrr = feed * depth * width / 1000.0 if cutting else 0.0
            torque = 0.2 * mrr + 0.03 * math.sin(theta)
            sim_rows.append([block, f"{x:.5f}", f"{y:.5f}", f"{z:.5f}", feed, 8000, f"{mrr:.6f}", f"{torque:.6f}", f"{depth:.4f}", f"{width:.4f}"])

            ax = x + rng.uniform(-0.015, 0.015)
            ay = y + rng.uniform(-0.015, 0.015)
            az = z + rng.uniform(-0.005, 0.005)
            act_f = feed * (0.95 + 0.03 * math.sin(theta + 0.7))
            load = 2.0 + 18.0 * mrr + 0.02 * act_f + 0.5 * math.sin(theta) + rng.uniform(-0.08, 0.08)
            if not cutting:
                load = 1.0 + rng.uniform(-0.05, 0.05)
            act_rows.append([block, "G01", f"{ax:.5f}", f"{ay:.5f}", f"{az:.5f}", 8000, feed, 7950, f"{act_f:.5f}", f"{load:.6f}", 60.0])

        sim_name = f"Sim_F{feed}_D{depth:g}.csv"
        act_name = f"Act_F{feed}_D{depth:g}_run{pair_idx + 1}.csv"
        _write_csv_rows(sim_dir / sim_name, ["block No", "orgX", "orgY", "orgZ", "OrgF", "OrgS", "MRR", "maxTorque", "Depth", "Width"], sim_rows)
        _write_csv_rows(act_dir / act_name, ["Block No", "G-code", "x", "y", "z", "command RPM", "command Feedrate", "actual RPM", "actual Feedrate", "actLoad", "Sampling Interval"], act_rows)
    return out_dir


def _runtime_metadata() -> dict[str, object]:
    import numpy as np
    import torch

    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "seed": 42,
    }


def _load_old_module():
    path = ROOT / "src" / "before" / "old.py"
    spec = importlib.util.spec_from_file_location("old_before", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["old_before"] = module
    spec.loader.exec_module(module)
    return module


def _measure(func: Callable[[], object]) -> tuple[float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        func()
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return elapsed, peak / 1e6


def _before_mapping(size: str, data_root: Path) -> int:
    old = _load_old_module()
    sim_glob = str(data_root / size / "Sim_20mm" / "*.csv")
    act_glob = str(data_root / size / "Act_20mm" / "*.csv")
    pairs = old.find_pairs_by_feed_depth(old.PathsConfig(sim_glob=sim_glob, act_glob=act_glob))
    total = 0
    for _, sim_path, act_path in pairs:
        sim = old.load_sim(sim_path)
        run = old.build_run_from_files(sim, act_path, old.MappingConfig(), old.LookAheadConfig())
        total += int(run.y_raw.shape[0])
    return total


def _after_mapping(size: str, data_root: Path) -> int:
    sim_glob = str(data_root / size / "Sim_20mm" / "*.csv")
    act_glob = str(data_root / size / "Act_20mm" / "*.csv")
    resolver = PathResolver(sim_glob, act_glob)
    mapper = ActMapper(MappingConfig())
    loader = SimLoader()
    total = 0
    for pair in resolver.iter_pairs():
        run = mapper.build_run(pair.name, loader.load(pair.sim_path), pair.act_path)
        total += int(run.y_raw.shape[0])
    return total


def _after_pipeline(size: str, data_root: Path) -> dict[str, object]:
    cfg = ExperimentConfig(
        paths=PathConfig(
            sim_glob=str(data_root / size / "Sim_20mm" / "*.csv"),
            act_glob=str(data_root / size / "Act_20mm" / "*.csv"),
            output_dir=ROOT / "results" / "after" / size,
        ),
        train=TrainConfig(epochs=1, hidden_size=16, tbptt_steps=64),
    )
    return ExperimentRunner(cfg).run()


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    metadata = _runtime_metadata()
    for row in rows:
        grouped.setdefault((str(row["size"]), str(row["variant"])), []).append(row)
    out: list[dict[str, object]] = []
    for (size, variant), items in grouped.items():
        elapsed = [float(x["elapsed_sec"]) for x in items]
        mem = [float(x["peak_memory_mb"]) for x in items]
        summary_row = {
            "size": size,
            "variant": variant,
            "repeat": len(items),
            "elapsed_sec_mean": statistics.fmean(elapsed),
            "elapsed_sec_std": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
            "peak_memory_mb_mean": statistics.fmean(mem),
            "peak_memory_mb_std": statistics.stdev(mem) if len(mem) > 1 else 0.0,
            "rows": items[0]["rows"],
        }
        summary_row.update(metadata)
        out.append(summary_row)
    by_size = {}
    for row in out:
        by_size.setdefault(row["size"], {})[row["variant"]] = row
    for variants in by_size.values():
        before = variants.get("before_mapping")
        after = variants.get("after_mapping")
        if before and after:
            speedup = float(before["elapsed_sec_mean"]) / max(float(after["elapsed_sec_mean"]), 1e-12)
            mem_reduction = 1.0 - float(after["peak_memory_mb_mean"]) / max(float(before["peak_memory_mb_mean"]), 1e-12)
            after["speedup_vs_before"] = speedup
            after["memory_reduction_vs_before"] = mem_reduction
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark before/after CNC pipeline code.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    repeat = max(1, int(args.repeat))
    raw_rows: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="cutload_benchmark_") as tmp:
        data_root = Path(tmp) / "synthetic"
        for size in SIZES:
            generate_synthetic(size, data_root)
            variants: list[tuple[str, Callable[[], object]]] = [
                ("before_mapping", lambda s=size: _before_mapping(s, data_root)),
                ("after_mapping", lambda s=size: _after_mapping(s, data_root)),
            ]
            if args.mode == "full":
                variants.append(("after_pipeline", lambda s=size: _after_pipeline(s, data_root)))
            for variant, func in variants:
                for i in range(repeat):
                    holder: dict[str, object] = {}
                    def wrapped() -> object:
                        result = func()
                        holder["result"] = result
                        return result
                    elapsed, mem = _measure(wrapped)
                    rows_count = holder["result"] if isinstance(holder.get("result"), int) else ""
                    raw_rows.append({"size": size, "variant": variant, "iteration": i + 1, "elapsed_sec": elapsed, "peak_memory_mb": mem, "rows": rows_count})
    summary = _summarize(raw_rows)
    _write_csv(ROOT / "results" / "benchmark_results.csv", summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
