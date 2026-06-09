from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import MappingConfig
from .csv_table import CsvTable
from .data_models import RunData, SimData
from .sim_loader import safe_float, safe_int

ACT_HEADERS_REQUIRED = ["Block No", "G-code", "x", "y", "z", "command RPM", "command Feedrate", "actual RPM", "actual Feedrate", "actLoad"]


def gcode_to_float(value: str) -> float:
    found = re.findall(r"G\s*0*([0-9]+)", str(value).upper())
    if found:
        return float(found[0])
    return safe_float(value, 0.0)


@dataclass(slots=True)
class BlockPointerState:
    last_local_idx: dict[int, int] = field(default_factory=dict)


class ActMapper:
    def __init__(self, cfg: MappingConfig) -> None:
        self.cfg = cfg

    def _nearest(self, sim: SimData, block_no: int, point: np.ndarray, state: BlockPointerState) -> tuple[np.ndarray, float] | None:
        idxs = sim.block_indices.get(block_no)
        if idxs is None or idxs.size == 0:
            if self.cfg.allow_missing_block:
                return None
            raise KeyError(f"Missing SIM block {block_no}")
        last = max(0, min(int(state.last_local_idx.get(block_no, 0)), len(idxs) - 1))
        start = max(0, last - self.cfg.window)
        end = min(len(idxs), last + self.cfg.window + 1)
        cand = idxs[start:end]
        diff = sim.xyz[cand] - point.reshape(1, 3)
        diff[:, 2] *= float(self.cfg.wz)
        best_local = int(np.argmin(np.sum(diff * diff, axis=1)))
        state.last_local_idx[block_no] = start + best_local
        idx = int(cand[best_local])
        return sim.feats[idx].copy(), float(sim.s[idx])

    @staticmethod
    def _lookahead(sim: SimData, s_target: float) -> np.ndarray:
        idx = int(np.searchsorted(sim.s, np.float32(s_target), side="left"))
        idx = max(0, min(idx, len(sim.s) - 1))
        return sim.feats[idx].copy()

    def build_run(self, name: str, sim: SimData, act_csv_path: Path, min_rows: int = 10) -> RunData:
        table = CsvTable(Path(act_csv_path), ACT_HEADERS_REQUIRED, max_scan_rows=40)
        state = BlockPointerState()
        recent_d: deque[float] = deque(maxlen=self.cfg.lookahead_steps)
        prev_point: np.ndarray | None = None
        prev_act_f: float | None = None
        mapped_rows: list[np.ndarray] = []
        look_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        mask_rows: list[bool] = []
        t_rows: list[float] = []
        t_ms = 0.0
        prev_dt = 60.0
        for row in table.iter_rows():
            dt = safe_float(table.get(row, "Sampling Interval"), prev_dt)
            if not np.isfinite(dt) or dt <= 0.0:
                dt = prev_dt
            prev_dt = dt
            t_ms += dt
            block_no = safe_int(table.get(row, "Block No"))
            point = np.array([safe_float(table.get(row, "x")), safe_float(table.get(row, "y")), safe_float(table.get(row, "z"))], dtype=np.float32)
            load = safe_float(table.get(row, "actLoad"))
            if block_no < 0 or np.isnan(point).any() or np.isnan(load):
                continue
            if prev_point is not None:
                d = float(np.linalg.norm(point - prev_point))
                if d > 0:
                    recent_d.append(d)
            prev_point = point
            mapped = self._nearest(sim, block_no, point, state)
            if mapped is None:
                continue
            sim_cur, s_at_p = mapped
            delta = float(sum(recent_d) / len(recent_d)) if recent_d else 1.0
            sim_look = self._lookahead(sim, s_at_p + delta)
            cmd_rpm = safe_float(table.get(row, "command RPM"), 0.0)
            cmd_f = safe_float(table.get(row, "command Feedrate"), 0.0)
            act_rpm = safe_float(table.get(row, "actual RPM"), 0.0)
            act_f = safe_float(table.get(row, "actual Feedrate"), 0.0)
            feed_err = act_f - cmd_f
            feed_err_ratio = feed_err / max(abs(cmd_f), 1.0)
            d_act_f = 0.0 if prev_act_f is None else act_f - prev_act_f
            prev_act_f = act_f
            onm = np.array([gcode_to_float(table.get(row, "G-code")), cmd_rpm, cmd_f, act_rpm, act_f, feed_err, feed_err_ratio, d_act_f], dtype=np.float32)
            mapped_rows.append(np.concatenate([onm, sim_cur, np.array([load], dtype=np.float32)]).astype(np.float32))
            look_rows.append(sim_look.astype(np.float32))
            y_rows.append(float(load))
            mask_rows.append(bool(sim.train_start_s <= s_at_p <= sim.train_end_s))
            t_rows.append(float(t_ms))
        if len(y_rows) < min_rows:
            raise ValueError(f"Too few mapped ACT rows: {act_csv_path} -> {len(y_rows)}")
        return RunData(name, np.stack(mapped_rows), np.stack(look_rows), np.array(y_rows, dtype=np.float32).reshape(-1, 1), np.array(mask_rows, dtype=np.bool_), np.array(t_rows, dtype=np.float32))
