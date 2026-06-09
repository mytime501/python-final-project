from __future__ import annotations

from pathlib import Path

import numpy as np

from .csv_table import CsvTable
from .data_models import SimData

SIM_HEADERS_REQUIRED = ["block No", "orgX", "orgY", "orgZ", "OrgF", "OrgS", "MRR", "maxTorque", "Depth", "Width"]


def safe_float(value: str, default: float = np.nan) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_int(value: str, default: int = -1) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


class SimLoader:
    def load(self, sim_csv_path: Path) -> SimData:
        table = CsvTable(Path(sim_csv_path), SIM_HEADERS_REQUIRED, max_scan_rows=20)
        blocks: list[int] = []
        xyz_rows: list[list[float]] = []
        base_rows: list[list[float]] = []
        for row in table.iter_rows():
            blocks.append(safe_int(table.get(row, "block No")))
            xyz_rows.append([safe_float(table.get(row, "orgX")), safe_float(table.get(row, "orgY")), safe_float(table.get(row, "orgZ"))])
            base_rows.append([
                safe_float(table.get(row, "OrgF"), 0.0),
                safe_float(table.get(row, "OrgS"), 0.0),
                safe_float(table.get(row, "Depth"), 0.0),
                safe_float(table.get(row, "MRR"), 0.0),
                safe_float(table.get(row, "maxTorque"), 0.0),
                safe_float(table.get(row, "Width"), 0.0),
            ])
        if not blocks:
            raise ValueError(f"Empty SIM CSV: {sim_csv_path}")
        block = np.array(blocks, dtype=np.int32)
        xyz = np.array(xyz_rows, dtype=np.float32)
        base = np.nan_to_num(np.array(base_rows, dtype=np.float32))
        s = np.zeros(len(block), dtype=np.float32)
        for i in range(1, len(block)):
            s[i] = s[i - 1] + float(np.linalg.norm(xyz[i] - xyz[i - 1]))
        mrr = base[:, 3]
        entry = np.zeros(len(block), dtype=np.float32)
        entry[1:] = ((mrr[:-1] <= 0.0) & (mrr[1:] > 0.0)).astype(np.float32)
        d_mrr = np.zeros(len(block), dtype=np.float32)
        d_mrr[1:] = mrr[1:] - mrr[:-1]
        feats = np.concatenate([base, entry[:, None], d_mrr[:, None]], axis=1).astype(np.float32)
        cut_idxs = np.where(mrr > 0.0)[0]
        if cut_idxs.size == 0:
            train_start_s, train_end_s = float(s[0]), float(s[-1])
        else:
            train_start_s, train_end_s = float(s[cut_idxs[0]]), float(s[cut_idxs[-1]])
        block_lists: dict[int, list[int]] = {}
        for i, b in enumerate(block):
            if b >= 0:
                block_lists.setdefault(int(b), []).append(i)
        block_indices = {b: np.array(idxs, dtype=np.int32) for b, idxs in block_lists.items()}
        return SimData(block, xyz, feats, s, block_indices, train_start_s, train_end_s)
