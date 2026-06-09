from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger("cutload_pipeline")


@dataclass(frozen=True, slots=True)
class FilePair:
    name: str
    sim_path: Path
    act_path: Path


class PathResolver:
    def __init__(self, sim_glob: str, act_glob: str) -> None:
        self.sim_glob = sim_glob
        self.act_glob = act_glob

    @staticmethod
    def _key(path: Path) -> str:
        name = path.name
        match = re.search(r"(?:Sim|Act)_F(\d+)_D([0-9]+(?:\.[0-9]+)?)", name, flags=re.IGNORECASE)
        if match:
            depth = match.group(2).rstrip("0").rstrip(".")
            return f"F{match.group(1)}_D{depth}"
        stem = path.stem.lower()
        nums = re.findall(r"\d+", stem)
        return nums[-1] if nums else stem.replace("_sim", "")

    def iter_pairs(self) -> Iterator[FilePair]:
        sim_paths = [Path(p) for p in glob.glob(self.sim_glob)]
        act_paths = [Path(p) for p in glob.glob(self.act_glob)]
        act_by_key: dict[str, Path] = {self._key(p): p for p in act_paths}
        for sim_path in sorted(sim_paths):
            key = self._key(sim_path)
            act_path = act_by_key.get(key)
            if act_path is not None:
                yield FilePair(key, sim_path, act_path)

    def resolve_target_cases(self, script_dir: Path) -> list[FilePair]:
        local = {
            "O80": (script_dir / "test1" / "20mm_80_2_sim.csv", script_dir / "test1" / "20mm_80_2_2026_01_21_15_01_39_654(O80, S80, T3_1).csv"),
            "O81": (script_dir / "test1" / "20mm_81_sim.csv", script_dir / "test1" / "20mm_81_2026_01_21_13_44_35_206(O81, S81, T3_1).csv"),
        }
        out: list[FilePair] = []
        for name in ("O80", "O81"):
            sim_path, act_path = local[name]
            if not (sim_path.exists() and act_path.exists()):
                LOGGER.warning("skip missing local target %s", name)
                continue
            out.append(FilePair(name, sim_path, act_path))
        return out
