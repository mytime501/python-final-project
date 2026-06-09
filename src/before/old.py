# train_stateful_gru_from_sim_act_pairs_v5_delta_depth.py
# ------------------------------------------------------------
# ✅ 변경점(핵심)
# 1) best 모델 선택 기준
#   - validation RMSE 기준 (기존과 동일)
#   - 단, persist margin loss는 고려하지 않음 (순수 RMSE 성능에 집중)
# 2) 파인튜닝
# ------------------------------------------------------------

import os
import re
import csv
import glob
import json
import datetime
import time
import shutil
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
# ============================================================
# CPU optimization defaults (4 cores)
# - Works even when you just run: python train_stateful_gru_tau_v3.py
# - You can override by setting env vars before launching.
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import torch
# ---- torch CPU threading ----
try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
except Exception:
    torch.set_num_threads(8)
try:
    # Inter-op threads: often best as 1 on small-core CPUs
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    torch.backends.mkldnn.enabled = True
except Exception:
    pass

import torch.nn as nn


# ============================================================
# Zero-arg 실행 지원 (중요)
# - `python train_stateful_gru_v32_tau.py` 만 실행해도 동작하도록
# - Sim_20mm/Act_20mm (또는 Sim_12mm/Act_12mm)를 스크립트 위치 기준으로 자동 탐색
# - 출력(onnx/scaler/debug)은 스크립트 위치에 고정 저장
# ============================================================

def _SCRIPT_DIR() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _resolve_default_globs() -> Tuple[str, str, str]:
    """Return (sim_glob, act_glob, tag). Prefer script-dir relative folders."""
    sd = _SCRIPT_DIR()

    def _g(base: str, sub: str) -> str:
        return os.path.join(base, sub, "*.csv")

    # 1) script folder (preferred)
    candidates = [
        ("Sim_20mm", "Act_20mm", "script/20mm"),
        ("Sim_12mm", "Act_12mm", "script/12mm"),
        (os.path.join("learning_data", "Sim_20mm"), os.path.join("learning_data", "Act_20mm"), "script/learning_data/20mm"),
        (os.path.join("learning_data", "Sim_12mm"), os.path.join("learning_data", "Act_12mm"), "script/learning_data/12mm"),
    ]
    for sdir, adir, tag in candidates:
        sg = _g(sd, sdir)
        ag = _g(sd, adir)
        if glob.glob(sg) and glob.glob(ag):
            return sg, ag, tag

    # 2) current working directory (fallback)
    cwd = os.getcwd()
    candidates2 = [
        ("Sim_20mm", "Act_20mm", "cwd/20mm"),
        ("Sim_12mm", "Act_12mm", "cwd/12mm"),
        (os.path.join("learning_data", "Sim_20mm"), os.path.join("learning_data", "Act_20mm"), "cwd/learning_data/20mm"),
        (os.path.join("learning_data", "Sim_12mm"), os.path.join("learning_data", "Act_12mm"), "cwd/learning_data/12mm"),
    ]
    for sdir, adir, tag in candidates2:
        sg = _g(cwd, sdir)
        ag = _g(cwd, adir)
        if glob.glob(sg) and glob.glob(ag):
            return sg, ag, tag

    # 3) last resort: default to script 20mm (may be empty; main will raise with guidance)
    return _g(sd, "Sim_20mm"), _g(sd, "Act_20mm"), "default(script/20mm)"


# ============================================================
# Hard-coded O80/O81 test-set paths
# - The script will try these fixed Windows paths first.
# - If they are not found, it falls back to ./test under the script folder.
# ============================================================
_HARDCODED_TEST_FILE_NAMES = {
    "O80": (
        "20mm_80_2_sim.csv",
        "20mm_80_2_2026_01_21_15_01_39_654(O80, S80, T3_1).csv",
    ),
    "O81": (
        "20mm_81_sim.csv",
        "20mm_81_2026_01_21_13_44_35_206(O81, S81, T3_1).csv",
    ),
}

_HARDCODED_TEST_CASE_CANDIDATES = {
    "O80": [
        (
            r"C:\Users\sg1\Desktop\train\test1\20mm_80_2_sim.csv",
            r"C:\Users\sg1\Desktop\train\test1\20mm_80_2_2026_01_21_15_01_39_654(O80, S80, T3_1).csv",
        ),
    ],
    "O81": [
        (
            r"C:\Users\sg1\Desktop\train\test1\20mm_81_sim.csv",
            r"C:\Users\sg1\Desktop\train\test1\20mm_81_2026_01_21_13_44_35_206(O81, S81, T3_1).csv",
        ),
    ],
}


def _resolve_hardcoded_test_cases(script_dir: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    test_dir = os.path.join(script_dir, "test1")

    for case_name in ("O80", "O81"):
        candidates = list(_HARDCODED_TEST_CASE_CANDIDATES.get(case_name, []))
        sim_name, act_name = _HARDCODED_TEST_FILE_NAMES[case_name]
        candidates.append((os.path.join(test_dir, sim_name), os.path.join(test_dir, act_name)))

        chosen = None
        for sim_p, act_p in candidates:
            if os.path.exists(sim_p) and os.path.exists(act_p):
                chosen = (sim_p, act_p)
                break
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is not None:
            out.append((case_name, chosen[0], chosen[1]))
    return out




# =========================
# 0) Configuration
# =========================

@dataclass
class PathsConfig:
    sim_glob: str = "Sim_20mm/*.csv"
    act_glob: str = "Act_20mm/*.csv"


@dataclass
class MappingConfig:
    window: int = 300
    wz: float = 3.0
    eps: float = 1e-12
    allow_missing_block: bool = True
    interp_max_seg_len: float = 1.5
    interp_max_proj_dist: float = 0.30
    interp_same_cut_state_only: bool = True
    interp_mrr_rel_jump_max: float = 0.30
    interp_depth_rel_jump_max: float = 0.30
    interp_width_rel_jump_max: float = 0.30


@dataclass
class LookAheadConfig:
    m: int = 10
    delta_min: float = 0.05
    delta_max: float = 10.0
    # Conservative future-command mixing to reduce false anticipation
    cmd_w_now: float = 0.85
    cmd_w_f1: float = 0.10
    cmd_w_f2: float = 0.05
    # Ramp-vs-fallback blending confidence
    blend_min_weight: float = 0.10
    track_err_ratio_ref: float = 0.20
    ramp_change_ref: float = 250.0
    # Temporal smoothing of Δ to avoid frame-to-frame jumps
    delta_ema_alpha: float = 0.25   # current-step weight in EMA
    delta_step_limit: float = 0.15  # max allowed Δ change per step


@dataclass
class ResidualFeedbackConfig:
    W: int = 50
    ki: float = 0.01
    kp: float = 0.02
    clip_i: float = 0.1
    clip_p: float = 0.1
    use_mean: bool = True


@dataclass
class TrainConfig:
    # RNN backbone
    hidden_size: int = 128
    num_layers: int = 1

    # Optim
    lr: float = 7e-4
    weight_decay: float = 1e-4
    epochs: int = 200          # max epochs (early stopping will usually stop earlier)
    tbptt_steps: int = 160

    # Speed: train on a random sub-window of each run per epoch (0<r<=1)
    train_sample_ratio: float = 0.35
    grad_clip: float = 1.0

    # Head (FC)
    # Default to baseline head (96→96→1) for strongest results on current dataset.
    head_type: str = "residual"    # baseline | bottleneck | residual | linear
    head_dropout: float = 0.05
    # ===== v23b: make raw output 'persist-grade' =====
    use_alpha_gate: bool = True          # dy_used = sigmoid(alpha)*dy
    alpha_l1: float = 0.002               # encourage small alpha (more persistence)
    alpha_sup: float = 0.0              # supervise alpha to match |dy| (helps raw become persist-like)
    alpha_target_scale: float = 1.0      # in dy_norm units: |dy|>=scale -> alpha_target≈1
    alpha_warmup_epochs: int = 0        # ramp alpha_l1/dy_l2 from 0→full to avoid early collapse
    dy_l2: float = 0.002                 # encourage small dy magnitude
    dy_weight_k: float = 4.0             # emphasize change events
    dy_weight_clip: float = 4.5          # clip |dy_norm| for weighting
    dy_weight_pow: float = 1.2          # event emphasis curve (1.0 linear, >1 focuses larger |dy|)
    dy_clip_norm: float = 8.0             # clamp model output dy_norm to avoid spikes
    loss_type: str = "huber"            # "mse" or "huber"
    huber_delta: float = 1.0             # huber delta in dy_norm space
    force_dy_mean_zero: bool = True      # set dy_mean=0 in scaler
    # ===== raw >= persist helpers (optional) =====
    persist_margin_lambda: float = 0.45  # penalize being worse than persist (in dy_norm SSE space)
    persist_margin_warmup_epochs: int = 12   # ramp margin loss over first N epochs
    persist_margin: float = 0.00         # margin (>=0). 0 means 'don't be worse than persist'
    dy_deadband_raw: float = 0.010         # set |dy_raw|<deadband to 0 in target (stabilizes raw)
    bottleneck_dim: int = 96

    # Early stopping (based on validation RMSE)
    early_stop: bool = True
    es_patience: int = 20
    es_min_delta: float = 1e-4
    es_eval_every: int = 5          # evaluate every N epochs
    # Best model selection (validation)
    # - "val_rmse": classic
    # - "r2_vs_persist": maximize raw R2 while discouraging being worse than persistence baseline
    best_metric: str = "r2_vs_persist"   # "val_rmse" | "r2_vs_persist"
    best_persist_penalty: float = 2.0   # penalty weight when raw R2 < persist R2
    best_persist_margin: float = 0.0    # allow raw R2 to be (persist - margin) without penalty
    es_min_delta_score: float = 1e-4    # min score improvement for best_metric="r2_vs_persist"
    # Best focus (which subset metric is used for best_metric)
    # - "cut": all validated samples (mask[t] & mask[t+1])  (default)
    # - "event": only samples with |dy_true_norm| >= best_event_dy_thresh_norm
    best_focus: str = "event"                 # "cut" | "event"
    best_event_dy_thresh_norm: float = 0.35  # event threshold in normalized dy units
    best_event_weight: float = 0.25         # tie-break / bonus weight using event gain (raw - persist) on events

    # Additional raw-stabilization losses (help raw approach persist without collapsing event response)
    stable_lambda: float = 0.015             # penalize dy_hat on stable segments
    stable_dy_thresh_norm: float = 0.22     # stable if |dy_true_norm| < thresh
    aircut_lambda: float = 0.05             # penalize dy_hat when MRR≈0 (air-cut)
    aircut_mrr_eps: float = 1e-9            # threshold on raw MRR

    # Finetune stage (2-phase training)
    finetune: bool = True
    ft_epochs: int = 160
    ft_lr_scale: float = 0.08
    ft_weight_decay_scale: float = 0.4
    ft_dy_deadband_scale: float = 0.20
    ft_persist_margin_lambda_scale: float = 0.7
    ft_dy_weight_k_scale: float = 1.4
    ft_dy_clip_norm_scale: float = 1.0
    ft_stable_lambda_scale: float = 0.6
    ft_aircut_lambda_scale: float = 0.8
    ft_patience: int = 20
    ft_eval_every: int = 5



    # ===== v26: target(O81) raw-max support =====
    # Runtime parity: reset states on non-cut/aircut segments so training/eval matches C# inference.
    reset_on_aircut: bool = True
    reset_on_aircut_hold_persist: bool = True  # (eval only) aircut -> y_pred = y_prev
    aircut_reset_min_steps: int = 1  # reset only after N consecutive non-cut steps (1 = immediate)


    # Best selection extensions
    # - val_rmse: minimize validation RMSE
    # - r2_vs_persist: maximize score on validation
    # - target_r2_vs_persist: maximize score on target run (O81)
    # - mix_r2: weighted mix of val-score and target-score
    target_score_weight: float = 0.70  # used only when best_metric == "mix_r2"

    # Validation calibration (when NO target run is provided)
    # - Helps raw R2 by analytically calibrating dy_gain/dy_bias on validation set.
    val_calib: bool = True

    # Target calibration / finetune (when target run is provided)
    target_calib: bool = True
    target_calib_only: bool = True   # True: only train dy_gain/dy_bias (analytic or tiny fit)
    target_calib_epochs: int = 120
    target_calib_lr_scale: float = 0.20
    target_calib_gain_min: float = 0.01  # allow strong shrink (post-like)
    target_calib_gain_max: float = 2.0
    target_calib_bias_abs: float = 2.0
    target_calib_weight_decay_scale: float = 0.0

    target_finetune: bool = True
    target_ft_epochs: int = 80
    target_ft_lr_scale: float = 0.05
    target_ft_weight_decay_scale: float = 0.2
    target_freeze_gru: bool = True

    # Runtime
    device: str = "cpu"


# =========================
# 1) CSV parsing helpers
# =========================

def norm_col(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def detect_delimiter(path: str, max_lines: int = 6) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = []
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip("\n\r")
            if line.strip() == "":
                continue
            lines.append(line)
        sample = "\n".join(lines)

    if not sample:
        return ","

    tab_count = sample.count("\t")
    comma_count = sample.count(",")
    if tab_count >= comma_count and tab_count > 0:
        return "\t"
    if comma_count > 0:
        return ","
    return "\t"


def find_header_row_index(path: str, required_cols: List[str], delimiter: str, max_scan_rows: int = 40) -> int:
    req = set(norm_col(c) for c in required_cols)
    with open(path, "r", newline="", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            if idx >= max_scan_rows:
                break
            if not row:
                continue
            cols = set(norm_col(c) for c in row if str(c).strip() != "")
            if req.issubset(cols):
                return idx
    raise ValueError(f"Header row not found in first {max_scan_rows} lines: {path}")


class CsvTable:
    def __init__(self, path: str, required_cols: List[str], max_scan_rows: int = 40):
        self.path = path
        self.delim = detect_delimiter(path)
        self.header_row = find_header_row_index(path, required_cols, self.delim, max_scan_rows=max_scan_rows)

        with open(path, "r", newline="", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f, delimiter=self.delim)
            header = None
            for i, row in enumerate(reader):
                if i == self.header_row:
                    header = row
                    break
        if header is None:
            raise ValueError(f"Failed to load header row: {path}")

        self.header = header
        self.col_to_idx: Dict[str, int] = {}
        for i, name in enumerate(header):
            key = norm_col(name)
            if key and key not in self.col_to_idx:
                self.col_to_idx[key] = i

        missing = [c for c in required_cols if norm_col(c) not in self.col_to_idx]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

    def iter_rows(self):
        with open(self.path, "r", newline="", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f, delimiter=self.delim)
            for i, row in enumerate(reader):
                if i <= self.header_row:
                    continue
                if not row:
                    continue
                yield row

    def get(self, row: List[str], col_name: str) -> str:
        idx = self.col_to_idx.get(norm_col(col_name), None)
        if idx is None or idx >= len(row):
            return ""
        return row[idx]


def _safe_float(v: str, default: float = np.nan) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except:
        return default


def _safe_int(v: str, default: int = -1) -> int:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return int(float(s))
    except:
        return default


def _gcode_to_float(gcode: str) -> float:
    """Robust G-code parser.

    - Accepts: 'G01', 'G1', 'G01 X..', 'G01/G41', ' 1 ', '01', etc.
    - If multiple G-codes exist in a string, prioritizes motion codes (G0/G1/G2/G3).
    - Returns an integer-like float (e.g., 1.0 for G01).
    """
    if gcode is None:
        return 0.0
    s = str(gcode).strip().upper()
    if s == "":
        return 0.0

    # 1) Extract all G-codes in the string
    codes: List[float] = []
    for mm in re.finditer(r"G\s*0*([0-9]+(?:\.[0-9]+)?)", s):
        try:
            codes.append(float(mm.group(1)))
        except:
            pass

    if codes:
        # Prefer motion codes when present
        for target in (0.0, 1.0, 2.0, 3.0):
            if any(int(c) == int(target) for c in codes):
                return float(int(target))

        # Otherwise return the first code
        try:
            return float(int(codes[0]))
        except:
            return float(codes[0])

    # 2) Fallback: pure numeric string (e.g., '1', '01')
    m2 = re.match(r"^\s*0*([0-9]+(?:\.[0-9]+)?)\s*$", s)
    if m2:
        try:
            return float(int(float(m2.group(1))))
        except:
            try:
                return float(m2.group(1))
            except:
                return 0.0

    return 0.0



# =========================
# 2) Required columns
# =========================

ACT_HEADERS_REQUIRED = [
    "Block No",
    "G-code",
    "x", "y", "z",
    "command RPM", "command Feedrate",
    "actual RPM", "actual Feedrate",
    "actLoad",
]

SIM_HEADERS_REQUIRED = [
    "block No",
    "orgX", "orgY", "orgZ",
    "OrgF", "OrgS",
    "MRR",
    "maxTorque",
    "Depth",
    "Width",
]


# =========================
# 3) SIM data structure
# =========================

@dataclass
class SimData:
    block: np.ndarray
    xyz: np.ndarray
    feats: np.ndarray     # [OrgF, OrgS, Depth, MRR, maxTorque, entry_flag, dMRR]
    s: np.ndarray
    block_indices: Dict[int, np.ndarray]
    train_start_s: float
    train_end_s: float


def load_sim(sim_csv_path: str) -> SimData:
    tbl = CsvTable(sim_csv_path, SIM_HEADERS_REQUIRED, max_scan_rows=20)
    rows = list(tbl.iter_rows())
    N = len(rows)
    if N == 0:
        raise ValueError(f"Empty sim data: {sim_csv_path}")

    block = np.empty((N,), dtype=np.int32)
    xyz = np.empty((N, 3), dtype=np.float32)
    base = np.empty((N, 6), dtype=np.float32)  # OrgF, OrgS, Depth, MRR, maxTorque, Width

    for i, r in enumerate(rows):
        block[i] = _safe_int(tbl.get(r, "block No"), -1)
        xyz[i, 0] = _safe_float(tbl.get(r, "orgX"))
        xyz[i, 1] = _safe_float(tbl.get(r, "orgY"))
        xyz[i, 2] = _safe_float(tbl.get(r, "orgZ"))
        base[i, 0] = _safe_float(tbl.get(r, "OrgF"))
        base[i, 1] = _safe_float(tbl.get(r, "OrgS"))
        base[i, 2] = _safe_float(tbl.get(r, "Depth"))
        base[i, 3] = _safe_float(tbl.get(r, "MRR"))
        base[i, 4] = _safe_float(tbl.get(r, "maxTorque"))
        base[i, 5] = _safe_float(tbl.get(r, "Width"))

    # cumulative distance s (file order)
    s = np.zeros((N,), dtype=np.float32)
    for i in range(1, N):
        d = float(np.linalg.norm(xyz[i] - xyz[i - 1]))
        s[i] = s[i - 1] + np.float32(d)

    mrr = base[:, 3]

    entry_flag = np.zeros((N,), dtype=np.float32)
    for i in range(1, N):
        if mrr[i - 1] == 0.0 and mrr[i] > 0.0:
            entry_flag[i] = 1.0

    start_idx = None
    for i in range(1, N):
        if entry_flag[i] == 1.0:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"No cutting region (MRR>0) found: {sim_csv_path}")

    end_idx = None
    for i in range(N - 1, -1, -1):
        if mrr[i] > 0.0:
            end_idx = i
            break
    if end_idx is None or end_idx <= start_idx:
        raise ValueError(f"Invalid cutting region: {sim_csv_path}")

    train_start_s = float(s[start_idx])
    train_end_s = float(s[end_idx])

    dMRR = np.zeros((N,), dtype=np.float32)
    dMRR[1:] = mrr[1:] - mrr[:-1]

    feats = np.concatenate(
        [base, entry_flag.reshape(-1, 1), dMRR.reshape(-1, 1)],
        axis=1
    ).astype(np.float32)

    # block -> global indices
    block_indices: Dict[int, List[int]] = {}
    for i, b in enumerate(block):
        if b < 0:
            continue
        block_indices.setdefault(int(b), []).append(i)

    block_idx_np: Dict[int, np.ndarray] = {
        b: np.array(idxs, dtype=np.int32) for b, idxs in block_indices.items()
    }

    return SimData(
        block=block,
        xyz=xyz,
        feats=feats,
        s=s,
        block_indices=block_idx_np,
        train_start_s=train_start_s,
        train_end_s=train_end_s
    )


# =========================
# 4) C# mapping logic (ported, nearest-point mapping without linear interpolation)
# =========================

@dataclass
class BlockPointerState:
    last_local_idx: Dict[int, int]


def _weighted_d2(ax, ay, az, x, y, z, wz: float) -> float:
    dx = x - ax
    dy = y - ay
    dz = (z - az) * wz
    return dx * dx + dy * dy + dz * dz

def _rel_diff(a: float, b: float, eps: float = 1e-9) -> float:
    return abs(a - b) / max(max(abs(a), abs(b)), eps)


def _same_cut_state(f0: np.ndarray, f1: np.ndarray, mrr_eps: float = 1e-9) -> bool:
    return bool((float(f0[3]) > mrr_eps) == (float(f1[3]) > mrr_eps))


def _blend_sim_feats(f0: np.ndarray, f1: np.ndarray, t: float) -> np.ndarray:
    out = f0.astype(np.float64).copy()
    cont_idx = [0, 1, 2, 3, 4, 5]
    out[cont_idx] = (1.0 - t) * f0[cont_idx] + t * f1[cont_idx]
    if t < 0.5:
        out[6] = f0[6]
        out[7] = f0[7]
    else:
        out[6] = f1[6]
        out[7] = f1[7]
    return out.astype(np.float32)


def _segment_ok_for_interp(sim: SimData, idx0: int, idx1: int, cfg: MappingConfig) -> bool:
    if idx0 == idx1:
        return False
    seg_len = abs(float(sim.s[idx1]) - float(sim.s[idx0]))
    if seg_len <= cfg.eps or seg_len > float(cfg.interp_max_seg_len):
        return False
    f0 = sim.feats[idx0]
    f1 = sim.feats[idx1]
    if bool(cfg.interp_same_cut_state_only) and (not _same_cut_state(f0, f1)):
        return False
    if float(f0[6]) > 0.5 or float(f1[6]) > 0.5:
        return False
    if _rel_diff(float(f0[3]), float(f1[3]), cfg.eps) > float(cfg.interp_mrr_rel_jump_max):
        return False
    if _rel_diff(float(f0[2]), float(f1[2]), cfg.eps) > float(cfg.interp_depth_rel_jump_max):
        return False
    if _rel_diff(float(f0[5]), float(f1[5]), cfg.eps) > float(cfg.interp_width_rel_jump_max):
        return False
    return True


def _pick_nearest_global(
    sim: SimData,
    idx: int,
    ax: float, ay: float, az: float,
    wz: float
) -> Tuple[float, np.ndarray, float]:
    x, y, z = sim.xyz[idx].astype(np.float64)
    d2 = _weighted_d2(ax, ay, az, float(x), float(y), float(z), wz)
    return float(d2), sim.feats[idx].copy(), float(sim.s[idx])


def map_act_to_sim_csharp(
    sim: SimData,
    bno: int,
    ax: float, ay: float, az: float,
    st: BlockPointerState,
    cfg: MappingConfig
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[Tuple[int, int, float]]]:
    idxs = sim.block_indices.get(bno, None)
    if idxs is None or idxs.size == 0:
        if cfg.allow_missing_block:
            return None, None, None
        raise KeyError(f"Missing sim block: {bno}")

    n = int(idxs.size)
    if n == 1:
        g = int(idxs[0])
        return sim.feats[g].copy(), float(sim.s[g]), (g, g, 0.0)

    last = st.last_local_idx.get(bno, 0)
    last = max(0, min(n - 1, last))

    start = max(0, last - cfg.window)
    end = min(n - 1, last + cfg.window)

    iBest = start
    bestD2 = float("inf")
    for li in range(start, end + 1):
        g = int(idxs[li])
        x, y, z = float(sim.xyz[g, 0]), float(sim.xyz[g, 1]), float(sim.xyz[g, 2])
        d2 = _weighted_d2(ax, ay, az, x, y, z, cfg.wz)
        if d2 < bestD2:
            bestD2 = d2
            iBest = li

    gBest = int(idxs[iBest])
    bestD2, bestFeat, bestS = _pick_nearest_global(sim, gBest, ax, ay, az, cfg.wz)
    st.last_local_idx[bno] = iBest
    return bestFeat, bestS, (gBest, gBest, 0.0)


def pick_lookahead_sim(sim: SimData, s_target: float, cfg: MappingConfig) -> np.ndarray:
    s = sim.s
    N = s.shape[0]
    if s_target <= float(s[0]):
        return sim.feats[0].copy()
    if s_target >= float(s[-1]):
        return sim.feats[-1].copy()

    j = int(np.searchsorted(s, np.float32(s_target), side="left"))
    j = max(1, min(N - 1, j))

    s0 = float(s[j - 1])
    s1 = float(s[j])
    if s1 == s0:
        return sim.feats[j].copy()

    if not _segment_ok_for_interp(sim, j - 1, j, cfg):
        if abs(s_target - s0) <= abs(s1 - s_target):
            return sim.feats[j - 1].copy()
        return sim.feats[j].copy()

    beta = (s_target - s0) / (s1 - s0)
    beta = max(0.0, min(1.0, beta))
    return _blend_sim_feats(sim.feats[j - 1], sim.feats[j], beta)


# =========================
# 5) Residual feedback (PI-like)
# =========================

class ResidualFeedbackState:
    def __init__(self, cfg: ResidualFeedbackConfig):
        self.cfg = cfg
        self.e_curr = 0.0
        self.e_hist = deque([0.0] * cfg.W, maxlen=cfg.W)

    def reset(self):
        self.e_curr = 0.0
        self.e_hist = deque([0.0] * self.cfg.W, maxlen=self.cfg.W)

    def compute_r(self) -> float:
        if self.cfg.W <= 0:
            mean_hist = 0.0
        else:
            mean_hist = float(sum(self.e_hist) / self.cfg.W) if self.cfg.use_mean else float(sum(self.e_hist))

        term_i = self.cfg.ki * mean_hist
        term_p = self.cfg.kp * self.e_curr

        term_i = max(-self.cfg.clip_i, min(self.cfg.clip_i, term_i))
        term_p = max(-self.cfg.clip_p, min(self.cfg.clip_p, term_p))
        return float(term_i + term_p)

    def update_with_new_residual(self, e_next: float):
        if self.cfg.W > 0:
            self.e_hist.appendleft(self.e_curr)
        self.e_curr = float(e_next)


# =========================
# 6) RunData + Scaler
# =========================

@dataclass
class RunData:
    mapped_raw: np.ndarray   # (N,15) raw
    look_raw: np.ndarray     # (N,6)  raw
    y_raw: np.ndarray        # (N,1)  raw actLoad
    train_mask: np.ndarray   # (N,) bool

    t_ms: np.ndarray        # (N,) cumulative time in ms (aligned to mapped rows)

    mapped: Optional[np.ndarray] = None  # normalized
    look: Optional[np.ndarray] = None    # normalized
    y: Optional[np.ndarray] = None       # normalized


    # torch caches (CPU tensors) to avoid re-creating tensors each epoch
    mapped_t: Optional[torch.Tensor] = None
    mapped_raw_t: Optional[torch.Tensor] = None
    look_t: Optional[torch.Tensor] = None
    y_raw_t: Optional[torch.Tensor] = None
    mask_t: Optional[torch.Tensor] = None


def ensure_run_tensors(run: RunData, device: torch.device) -> None:
    """Materialize and cache torch tensors for a run to avoid recreating them each epoch."""
    # Create CPU tensors once
    if getattr(run, "mapped_t", None) is None and getattr(run, "mapped", None) is not None:
        run.mapped_t = torch.from_numpy(run.mapped).to(dtype=torch.float32)
    if getattr(run, "mapped_raw_t", None) is None and getattr(run, "mapped_raw", None) is not None:
        run.mapped_raw_t = torch.from_numpy(run.mapped_raw).to(dtype=torch.float32)
    if getattr(run, "look_t", None) is None and getattr(run, "look", None) is not None:
        run.look_t = torch.from_numpy(run.look).to(dtype=torch.float32)
    if getattr(run, "y_raw_t", None) is None and getattr(run, "y_raw", None) is not None:
        run.y_raw_t = torch.from_numpy(run.y_raw).to(dtype=torch.float32)
    if getattr(run, "mask_t", None) is None and getattr(run, "train_mask", None) is not None:
        run.mask_t = torch.from_numpy(run.train_mask.astype(np.bool_))

    # Move to device (CPU no-op; keeps code consistent if CUDA is enabled later)
    if run.mapped_t is not None and run.mapped_t.device != device:
        run.mapped_t = run.mapped_t.to(device)
    if run.mapped_raw_t is not None and run.mapped_raw_t.device != device:
        run.mapped_raw_t = run.mapped_raw_t.to(device)
    if run.look_t is not None and run.look_t.device != device:
        run.look_t = run.look_t.to(device)
    if run.y_raw_t is not None and run.y_raw_t.device != device:
        run.y_raw_t = run.y_raw_t.to(device)
    if run.mask_t is not None and run.mask_t.device != device:
        run.mask_t = run.mask_t.to(device)



# =========================
# 6.5) SIM→Load delay compensation (τ-ms + interpolation)
# =========================

def _corr_coef(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float('-inf')
    a = a[m]; b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()) + eps)
    if denom <= eps:
        return float('-inf')
    return float((a*b).sum() / denom)

def _dt_median_ms(run: RunData, fallback: float = 60.0) -> float:
    try:
        t = run.t_ms.astype(np.float64).reshape(-1)
        if t.size < 3:
            return float(fallback)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if dt.size < 10:
            return float(fallback)
        return float(np.median(dt))
    except:
        return float(fallback)

def estimate_lag_samples_from_run(run: RunData,
                                  max_lag: int = 25,
                                  method: str = 'combo',
                                  mrr_eps: float = 1e-9,
                                  min_points: int = 500) -> Tuple[int, Dict[str, float]]:
    """Estimate lag L (samples) where SIM features lead actLoad.

    We search L in [0..max_lag] maximizing corr( sim_feat[t], actLoad[t+L] )
    over cutting points only.

    Returns: (best_lag, stats_dict)
    """
    max_lag = int(max(0, max_lag))
    if max_lag == 0:
        return 0, {'corr': 0.0}

    sim_dim = int(run.look_raw.shape[1])
    onm_dim = int(run.mapped_raw.shape[1] - sim_dim - 1)
    # sim_cur indices: [onm .. onm+sim_dim-1]
    mrr_idx = onm_dim + 3
    tq_idx = onm_dim + 4

    mrr = run.mapped_raw[:, mrr_idx].astype(np.float64)
    tq = run.mapped_raw[:, tq_idx].astype(np.float64)
    y = run.y_raw.reshape(-1).astype(np.float64)

    base_mask = run.train_mask.astype(bool) & (mrr > float(mrr_eps)) & np.isfinite(y)
    if base_mask.sum() < min_points:
        return 0, {'corr': 0.0, 'n': float(base_mask.sum())}

    meth = str(method).lower()
    use_diff = meth.endswith('_d1')
    if use_diff:
        meth = meth[:-3]
    if meth == 'mrr':
        x = mrr
    elif meth == 'torque':
        x = tq
    else:
        x = 0.5 * (mrr + tq)

    if use_diff:
        x = np.diff(x, prepend=x[0])
        y = np.diff(y, prepend=y[0])

    best_lag = 0
    best_c = float('-inf')
    for L in range(0, max_lag + 1):
        if L == 0:
            mask = base_mask
            xa = x
            ya = y
        else:
            mask = base_mask[:-L] & base_mask[L:]
            xa = x[:-L]
            ya = y[L:]
        if mask.sum() < min_points:
            continue
        c = _corr_coef(xa[mask], ya[mask])
        if c > best_c:
            best_c = c
            best_lag = L

    return int(best_lag), {'corr': float(best_c), 'n': float(base_mask.sum())}

def apply_sim_tau_to_run(run: RunData,
                         tau_ms: float,
                         mrr_eps: float = 1e-9) -> None:
    """Apply delay compensation by time (τ-ms) using interpolation.

    For each sample i at time t_i (ms), we replace ONLY SIM features with interpolated values at (t_i - τ).
    - on-machine features and actLoad(t) are unchanged.
    - train_mask is AND-ed with: in-range & aligned MRR>eps & finite.

    This is more robust than pure L-sample shift when dt jitters or some rows are skipped.
    """
    tau = float(max(0.0, tau_ms))
    if tau <= 0.0:
        return

    t = run.t_ms.astype(np.float64).reshape(-1)
    N = int(run.mapped_raw.shape[0])
    if N < 5 or t.size != N:
        run.train_mask[:] = False
        return

    # Ensure time is (mostly) increasing; if not, make it non-decreasing.
    for i in range(1, N):
        if t[i] < t[i-1]:
            t[i] = t[i-1]

    tq = t - tau

    sim_dim = int(run.look_raw.shape[1])
    onm_dim = int(run.mapped_raw.shape[1] - sim_dim - 1)
    sim_slice = slice(onm_dim, onm_dim + sim_dim)

    old_mapped = run.mapped_raw
    old_look = run.look_raw
    old_mask = run.train_mask.astype(bool)

    new_mapped = old_mapped.copy().astype(np.float32)
    new_look = old_look.copy().astype(np.float32)

    # interpolate sim_cur
    for j in range(sim_dim):
        v = old_mapped[:, onm_dim + j].astype(np.float64)
        new_mapped[:, onm_dim + j] = np.interp(tq, t, v, left=np.nan, right=np.nan).astype(np.float32)

    # interpolate sim_look
    for j in range(sim_dim):
        v = old_look[:, j].astype(np.float64)
        new_look[:, j] = np.interp(tq, t, v, left=np.nan, right=np.nan).astype(np.float32)

    # mask: require in-range and aligned MRR indicates cutting
    mrr_idx = onm_dim + 3
    aligned_mrr = new_mapped[:, mrr_idx].astype(np.float64)

    in_range = (tq >= float(t[0])) & (tq <= float(t[-1]))
    finite_sim = np.isfinite(aligned_mrr)
    new_mask = old_mask & in_range & finite_sim & (aligned_mrr > float(mrr_eps))

    run.mapped_raw = new_mapped
    run.look_raw = new_look
    run.train_mask = new_mask.astype(np.bool_)
@dataclass
class FeatureScaler:
    mapped_mean: np.ndarray  # (14,)
    mapped_std: np.ndarray   # (14,)
    look_mean: np.ndarray    # (6,)
    look_std: np.ndarray     # (6,)
    y_mean: float
    y_std: float
    dy_mean: float
    dy_std: float

    def apply_run(self, run: RunData):
        run.mapped = ((run.mapped_raw - self.mapped_mean) / self.mapped_std).astype(np.float32)
        run.look = ((run.look_raw - self.look_mean) / self.look_std).astype(np.float32)
        run.y = ((run.y_raw - self.y_mean) / self.y_std).astype(np.float32)

    def to_json(self) -> Dict:
        return {
            "mapped_mean": self.mapped_mean.tolist(),
            "mapped_std": self.mapped_std.tolist(),
            "look_mean": self.look_mean.tolist(),
            "look_std": self.look_std.tolist(),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "dy_mean": float(self.dy_mean),
            "dy_std": float(self.dy_std),
            "note": "Model predicts delta_y_norm for next step. Apply x_norm=(x-mean)/std exactly in C# before ONNX. delta_hat_raw = delta_hat_norm*dy_std + dy_mean; y_hat_next = y_prev + delta_hat_raw. (y_mean/y_std kept for reference only)"
        }

    @staticmethod
    def fit(runs: List[RunData], eps: float = 1e-8, force_dy_mean_zero: bool = True, stage_runs: Optional[List[RunData]] = None) -> "FeatureScaler":
        # 학습 가능한 구간만( mask[t] && mask[t+1] ) 기반으로 통계 계산
        mapped_list = []
        look_list = []
        y_list = []
        dy_list = []

        for run in (stage_runs if stage_runs is not None else runs):
            N = run.mapped_raw.shape[0]
            if N < 2:
                continue
            m = run.train_mask
            valid = np.logical_and(m[:-1], m[1:])  # t에서 next를 예측 가능한 구간
            if not np.any(valid):
                continue

            mapped_list.append(run.mapped_raw[:-1][valid])
            look_list.append(run.look_raw[:-1][valid])
            # y는 (t+1) 분포도 반영되게 next쪽도 포함
            y_list.append(run.y_raw[1:][valid])
            dy_list.append((run.y_raw[1:][valid] - run.y_raw[:-1][valid]).reshape(-1))

        X1 = np.concatenate(mapped_list, axis=0).astype(np.float64) if mapped_list else None
        X2 = np.concatenate(look_list, axis=0).astype(np.float64) if look_list else None
        Y = np.concatenate(y_list, axis=0).reshape(-1).astype(np.float64) if y_list else None

        if X1 is None or X2 is None or Y is None or Y.size < 10:
            raise RuntimeError("Scaler fit failed: too few valid training samples.")

        mapped_mean = X1.mean(axis=0)
        mapped_std = X1.std(axis=0)
        mapped_std = np.maximum(mapped_std, eps)

        look_mean = X2.mean(axis=0)
        look_std = X2.std(axis=0)
        look_std = np.maximum(look_std, eps)

        y_mean = float(Y.mean())
        y_std = float(Y.std())
        y_std = max(y_std, eps)

        DY = np.concatenate(dy_list, axis=0).astype(np.float64) if dy_list else None
        if DY is None or DY.size < 10:
            raise RuntimeError("Scaler fit failed: too few valid delta samples.")
        dy_mean = float(DY.mean())
        if force_dy_mean_zero:
            dy_mean = 0.0
        dy_std = float(DY.std())
        dy_std = max(dy_std, eps)

        return FeatureScaler(
            mapped_mean=mapped_mean.astype(np.float32),
            mapped_std=mapped_std.astype(np.float32),
            look_mean=look_mean.astype(np.float32),
            look_std=look_std.astype(np.float32),
            y_mean=y_mean,
            y_std=y_std,
            dy_mean=dy_mean,
            dy_std=dy_std
        )


def _first_cut_index(mask: np.ndarray) -> Optional[int]:
    if mask.size == 0:
        return None
    idx = int(np.argmax(mask.astype(np.int32)))
    if not bool(mask[idx]):
        return None
    return idx

def _predict_dt_ms_from_history(dt_hist: deque, default_dt_ms: float = 60.0) -> float:
    vals = [float(v) for v in list(dt_hist) if np.isfinite(v) and float(v) > 0.0]
    if not vals:
        return float(default_dt_ms)
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _weighted_future_command_feed(
    cmd_now: float,
    future_cmds: List[float],
    look_cfg: LookAheadConfig,
    future_scale: float = 1.0,
) -> float:
    vals = [float(cmd_now)] + [float(v) for v in future_cmds]
    fs = float(np.clip(float(future_scale), 0.0, 1.0))
    weights = [
        float(look_cfg.cmd_w_now),
        float(look_cfg.cmd_w_f1) * fs,
        float(look_cfg.cmd_w_f2) * fs,
    ]
    num = 0.0
    den = 0.0
    for i, v in enumerate(vals[:len(weights)]):
        if np.isfinite(v) and v >= 0.0:
            w = float(weights[i])
            if w <= 0.0:
                continue
            num += w * v
            den += w
    if den <= 1e-12:
        return float(cmd_now) if np.isfinite(cmd_now) else 0.0
    return float(num / den)


def _predict_next_actual_feedrate(
    actual_hist: deque,
    dt_hist_ms: deque,
    cmd_now: float,
    future_cmds: List[float],
    look_cfg: LookAheadConfig,
    future_scale: float = 1.0,
) -> Tuple[float, float, float]:
    act_vals = [float(v) for v in list(actual_hist) if np.isfinite(v) and float(v) >= 0.0]
    if act_vals:
        act_t = float(act_vals[-1])
    else:
        act_t = float(cmd_now) if np.isfinite(cmd_now) else 0.0

    dt_pred_ms = _predict_dt_ms_from_history(dt_hist_ms, default_dt_ms=60.0)
    cmd_future_ref = _weighted_future_command_feed(cmd_now, future_cmds, look_cfg=look_cfg, future_scale=future_scale)

    slope_vals: List[float] = []
    slope_wts: List[float] = []
    if len(act_vals) >= 2:
        dt_vals = [float(v) for v in list(dt_hist_ms) if np.isfinite(v) and float(v) > 0.0]
        n_pairs = min(len(act_vals) - 1, len(dt_vals))
        for k in range(n_pairs):
            i0 = len(act_vals) - n_pairs - 1 + k
            i1 = i0 + 1
            dt_k = max(float(dt_vals[len(dt_vals) - n_pairs + k]), 1e-9)
            slope = (float(act_vals[i1]) - float(act_vals[i0])) / dt_k
            slope_vals.append(float(slope))
            slope_wts.append(float(k + 1))

    if slope_vals and sum(slope_wts) > 0.0:
        slope_per_ms = float(np.average(np.asarray(slope_vals, dtype=np.float64), weights=np.asarray(slope_wts, dtype=np.float64)))
    else:
        slope_per_ms = 0.0

    trend_step = slope_per_ms * dt_pred_ms
    trend_gain = 0.40
    follow_gain = float(np.clip(dt_pred_ms / 120.0, 0.20, 0.60))

    act_pred = act_t + trend_gain * trend_step + follow_gain * (cmd_future_ref - act_t)

    max_step = max(100.0, abs(trend_step) * 2.0 + 0.60 * abs(cmd_future_ref - act_t))
    act_pred = float(np.clip(act_pred, act_t - max_step, act_t + max_step))
    act_pred = max(0.0, act_pred)

    return float(act_pred), float(dt_pred_ms), float(cmd_future_ref)


def _estimate_lookahead_delta_from_feed_ramp(
    actual_hist: deque,
    dt_hist_ms: deque,
    cmd_now: float,
    future_cmds: List[float],
    look_cfg: LookAheadConfig,
    fallback_delta: float,
    future_scale: float = 1.0,
) -> float:
    act_pred_next, dt_pred_ms, _ = _predict_next_actual_feedrate(
        actual_hist=actual_hist,
        dt_hist_ms=dt_hist_ms,
        cmd_now=float(cmd_now),
        future_cmds=future_cmds,
        look_cfg=look_cfg,
        future_scale=future_scale,
    )

    act_vals = [float(v) for v in list(actual_hist) if np.isfinite(v) and float(v) >= 0.0]
    act_t = float(act_vals[-1]) if act_vals else float(cmd_now)
    if (not np.isfinite(act_t)) or act_t < 0.0:
        act_t = 0.0

    if np.isfinite(dt_pred_ms) and dt_pred_ms > 0.0:
        avg_feed_over_interval = 0.5 * (act_t + act_pred_next)
        delta = avg_feed_over_interval * dt_pred_ms / 60000.0
    else:
        delta = float(fallback_delta)

    if (not np.isfinite(delta)) or delta <= 0.0:
        delta = float(fallback_delta)

    delta = max(float(look_cfg.delta_min), min(float(look_cfg.delta_max), float(delta)))
    return float(delta)


def _blend_lookahead_delta(
    ramp_delta: float,
    fallback_delta: float,
    feed_err_ratio: float,
    d_act_f: float,
    look_cfg: LookAheadConfig,
) -> float:
    if (not np.isfinite(ramp_delta)) or ramp_delta <= 0.0:
        ramp_delta = float(fallback_delta)

    tr_ref = max(float(look_cfg.track_err_ratio_ref), 1e-9)
    rc_ref = max(float(look_cfg.ramp_change_ref), 1e-9)

    track_conf = 1.0 - float(np.clip(abs(float(feed_err_ratio)) / tr_ref, 0.0, 1.0))
    ramp_conf = 1.0 - float(np.clip(abs(float(d_act_f)) / rc_ref, 0.0, 1.0))
    w = max(float(look_cfg.blend_min_weight), 0.60 * track_conf + 0.40 * ramp_conf)

    delta = float(w * float(ramp_delta) + (1.0 - w) * float(fallback_delta))
    delta = max(float(look_cfg.delta_min), min(float(look_cfg.delta_max), float(delta)))
    return float(delta)


def _smooth_lookahead_delta(
    delta: float,
    prev_delta: Optional[float],
    look_cfg: LookAheadConfig,
) -> float:
    cur = float(delta)
    if prev_delta is None or (not np.isfinite(prev_delta)):
        sm = cur
    else:
        alpha = float(np.clip(float(look_cfg.delta_ema_alpha), 0.0, 1.0))
        sm = (1.0 - alpha) * float(prev_delta) + alpha * cur

        step_lim = max(float(look_cfg.delta_step_limit), 0.0)
        if step_lim > 0.0:
            sm = float(np.clip(sm, float(prev_delta) - step_lim, float(prev_delta) + step_lim))

    sm = max(float(look_cfg.delta_min), min(float(look_cfg.delta_max), float(sm)))
    return float(sm)


# =========================
# 7) Build run from files
# =========================

def build_run_from_files(
    sim: SimData,
    act_csv_path: str,
    map_cfg: MappingConfig,
    look_cfg: LookAheadConfig,
) -> RunData:
    act_tbl = CsvTable(act_csv_path, ACT_HEADERS_REQUIRED, max_scan_rows=40)
    act_rows = list(act_tbl.iter_rows())

    st = BlockPointerState(last_local_idx={})
    prev_xyz: Optional[np.ndarray] = None
    recent_d = deque(maxlen=look_cfg.m)

    prev_act_f_val: Optional[float] = None
    prev_delta: Optional[float] = None
    actual_feed_hist = deque(maxlen=5)
    dt_hist_ms = deque(maxlen=5)

    mapped_rows: List[np.ndarray] = []
    look_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    mask_rows: List[bool] = []

    time_rows: List[float] = []
    t_ms = 0.0
    prev_dt = 60.0

    for row_idx, row in enumerate(act_rows):
        # time accumulation (ms): prefer 'Sampling Interval' column if present
        dt = _safe_float(act_tbl.get(row, "Sampling Interval"), np.nan)
        if not np.isfinite(dt):
            dt = prev_dt
        elif dt < 0.0:
            dt = prev_dt
        elif dt == 0.0 and t_ms <= 0.0:
            dt = 0.0
        elif dt == 0.0:
            dt = prev_dt
        prev_dt = float(dt)
        if np.isfinite(float(dt)) and float(dt) > 0.0:
            dt_hist_ms.append(float(dt))
        t_ms += float(dt)
        bno = _safe_int(act_tbl.get(row, "Block No"), -1)
        g = _gcode_to_float(act_tbl.get(row, "G-code"))

        ax = _safe_float(act_tbl.get(row, "x"))
        ay = _safe_float(act_tbl.get(row, "y"))
        az = _safe_float(act_tbl.get(row, "z"))

        cmd_rpm = _safe_float(act_tbl.get(row, "command RPM"))
        cmd_f = _safe_float(act_tbl.get(row, "command Feedrate"))
        act_rpm = _safe_float(act_tbl.get(row, "actual RPM"))
        act_f = _safe_float(act_tbl.get(row, "actual Feedrate"))

        # Guard against NaNs in command/actual signals (mapping uses xyz/load only)
        if np.isnan(cmd_rpm): cmd_rpm = 0.0
        if np.isnan(cmd_f): cmd_f = 0.0
        if np.isnan(act_rpm): act_rpm = 0.0
        if np.isnan(act_f): act_f = 0.0

        # Feed tracking dynamics: actual feed ramps -> affects cutting load
        feed_err = float(act_f - cmd_f)
        feed_err_ratio = float(feed_err / max(abs(cmd_f), 1.0))
        d_act_f = 0.0 if prev_act_f_val is None else float(act_f - prev_act_f_val)

        load = _safe_float(act_tbl.get(row, "actLoad"))

        if bno < 0 or np.isnan(ax) or np.isnan(ay) or np.isnan(az) or np.isnan(load):
            continue

        P = np.array([ax, ay, az], dtype=np.float32)

        if prev_xyz is not None:
            d = float(np.linalg.norm(P - prev_xyz))
            if not np.isnan(d) and d > 0:
                recent_d.append(d)
        prev_xyz = P

        actual_feed_hist.append(float(act_f))

        future_cmds: List[float] = []
        for j in range(1, 3):
            if (row_idx + j) < len(act_rows):
                cmd_f_j = _safe_float(act_tbl.get(act_rows[row_idx + j], "command Feedrate"), np.nan)
                if np.isfinite(cmd_f_j) and cmd_f_j >= 0.0:
                    future_cmds.append(float(cmd_f_j))

        fallback_delta = float(sum(recent_d) / len(recent_d)) if len(recent_d) else 1.0

        tr_ref = max(float(look_cfg.track_err_ratio_ref), 1e-9)
        rc_ref = max(float(look_cfg.ramp_change_ref), 1e-9)
        future_scale_track = 1.0 - float(np.clip(abs(feed_err_ratio) / tr_ref, 0.0, 1.0))
        future_scale_ramp = 1.0 - float(np.clip(abs(d_act_f) / rc_ref, 0.0, 1.0))
        future_scale = float(np.clip(0.70 * future_scale_track + 0.30 * future_scale_ramp, 0.0, 1.0))

        delta_ramp = _estimate_lookahead_delta_from_feed_ramp(
            actual_hist=actual_feed_hist,
            dt_hist_ms=dt_hist_ms,
            cmd_now=float(cmd_f),
            future_cmds=future_cmds,
            look_cfg=look_cfg,
            fallback_delta=fallback_delta,
            future_scale=future_scale,
        )
        delta_blend = _blend_lookahead_delta(
            ramp_delta=delta_ramp,
            fallback_delta=fallback_delta,
            feed_err_ratio=feed_err_ratio,
            d_act_f=d_act_f,
            look_cfg=look_cfg,
        )
        delta = _smooth_lookahead_delta(delta_blend, prev_delta=prev_delta, look_cfg=look_cfg)
        prev_delta = float(delta)

        sim_cur, s_at_P, _dbg = map_act_to_sim_csharp(sim, bno, ax, ay, az, st, map_cfg)
        if sim_cur is None or s_at_P is None:
            continue

        sim_look = pick_lookahead_sim(sim, float(s_at_P + delta), map_cfg)

        # ✅ initial/ending MRR=0 구간은 학습/평가 마스크 False
        in_train = (sim.train_start_s <= s_at_P <= sim.train_end_s)

        # on-machine features (xyz is only for mapping / Δ(lookahead) estimation)
        # - feed_err / feed_err_ratio: tracking error between commanded vs actual feed
        # - d_act_f: sequential change of actual feed (ramp rate proxy)
        prev_act_f_val = float(act_f)
        onm = np.array([g, cmd_rpm, cmd_f, act_rpm, act_f, feed_err, feed_err_ratio, d_act_f], dtype=np.float32)

        # ✅ actLoad(t) 추가 -> mapped(17) = onm(8) + sim_cur(8: +Width 포함) + actLoad_t(1)
        mapped = np.concatenate([onm, sim_cur.astype(np.float32), np.array([load], dtype=np.float32)], axis=0)

        mapped_rows.append(mapped.astype(np.float32))
        time_rows.append(float(t_ms))
        look_rows.append(sim_look.astype(np.float32))
        y_rows.append(float(load))
        mask_rows.append(bool(in_train))

    if len(y_rows) < 10:
        raise ValueError(f"Too few samples after mapping: {act_csv_path} -> {len(y_rows)}")

    return RunData(
        mapped_raw=np.stack(mapped_rows, axis=0).astype(np.float32),
        look_raw=np.stack(look_rows, axis=0).astype(np.float32),
        y_raw=np.array(y_rows, dtype=np.float32).reshape(-1, 1),
        train_mask=np.array(mask_rows, dtype=np.bool_),
        t_ms=np.array(time_rows, dtype=np.float32)
    )


# =========================
# 8) Model
# =========================

class GRUStepModel(nn.Module):
    """
    Stateful GRU + regression head for dy_norm.

    v12b goal: make *raw* model output behave closer to a persistence baseline (dy≈0)
    without relying on an external post-processor.

    - We add a learned gate alpha∈[0,1] and output dy_used = alpha * dy_base.
      When the model is uncertain, it can lower alpha → dy_used→0 (persistence).
    - We also clamp dy_norm (anti-spike) and apply regularization during training.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 head_type: str = "baseline",
                 head_dropout: float = 0.05,
                 bottleneck_dim: int = 32,
                 use_alpha_gate: bool = True):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.use_alpha_gate = bool(use_alpha_gate)

        # v26: lightweight output calibration (dy_norm space)
        self.dy_gain = nn.Parameter(torch.ones(1))
        self.dy_bias = nn.Parameter(torch.zeros(1))

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # If head_type is None, fall back to baseline (keeps behavior stable).
        self.head_type = (head_type or "baseline").lower()
        self.head_dropout = float(head_dropout)
        self.bottleneck_dim = int(bottleneck_dim)

        # Common normalization (helps regression stability)
        self.ln = nn.LayerNorm(hidden_size)

        if self.head_type == "baseline":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.head_dropout),
                nn.Linear(hidden_size, 1)
            )
        elif self.head_type == "bottleneck":
            b = max(4, self.bottleneck_dim)
            self.head = nn.Sequential(
                self.ln,
                nn.Linear(hidden_size, b),
                nn.GELU(),
                nn.Dropout(p=self.head_dropout),
                nn.Linear(b, 1)
            )
        elif self.head_type == "linear":
            self.head = nn.Sequential(
                self.ln,
                nn.Linear(hidden_size, 1)
            )
        elif self.head_type == "residual":
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.drop = nn.Dropout(p=self.head_dropout)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # alpha gate (simple + stable): LayerNorm + Linear -> sigmoid
        if self.use_alpha_gate:
            self.alpha_head = nn.Sequential(
                self.ln,
                nn.Linear(hidden_size, 1)
            )

    def _head_forward(self, h_last: torch.Tensor) -> torch.Tensor:
        if self.head_type in ("baseline", "bottleneck", "linear"):
            return self.head(h_last)
        # residual
        h = self.ln(h_last)
        z = torch.nn.functional.gelu(self.fc1(h))
        z = self.drop(z)
        return self.fc2(h + z)

    def _alpha_forward(self, h_last: torch.Tensor) -> torch.Tensor:
        # returns alpha in [0,1] with shape (B,1)
        if not self.use_alpha_gate:
            return torch.ones((h_last.shape[0], 1), device=h_last.device, dtype=h_last.dtype)
        return torch.sigmoid(self.alpha_head(h_last))

    def forward_with_alpha(self, x: torch.Tensor, h_in: torch.Tensor):
        # x: (B,F) or (B,1,F)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,F)
        out, h_out = self.gru(x, h_in)
        h_last = out[:, -1, :]
        dy_base = self._head_forward(h_last)         # (B,1)
        alpha = self._alpha_forward(h_last)          # (B,1)
        dy_used = dy_base * alpha                    # (B,1)
        # v26: apply affine calibration in dy_norm space
        dy_used = dy_used * self.dy_gain + self.dy_bias
        return dy_used, alpha, h_out

    def forward(self, x: torch.Tensor, h_in: torch.Tensor):
        dy_used, _, h_out = self.forward_with_alpha(x, h_in)
        return dy_used, h_out





# ==================
# v26+: Target(O81) dy affine calibration (analytic, safe keep/revert)
# ==================

def score_from_val(val_res: Dict, cfg: "TrainConfig") -> Tuple[float, float, float]:
    """Return (score, raw_r2, persist_r2) for checkpoint selection.

    - raw_r2 / persist_r2 are chosen by cfg.best_focus ("cut" or "event")
    - score = raw_r2 - penalty + bonus
      * penalty only applies when raw is worse than persist beyond cfg.best_persist_margin
      * bonus uses event gain (raw_event - persist_event) weighted by cfg.best_event_weight
    """
    raw_cut = float(val_res.get("torch", {}).get("R2", float("-inf")))
    persist_cut = float(val_res.get("persist_baseline", {}).get("R2", float("-inf")))

    raw_evt = float(val_res.get("torch_event", {}).get("R2", raw_cut))
    persist_evt = float(val_res.get("persist_event", {}).get("R2", persist_cut))

    if str(getattr(cfg, "best_focus", "cut")).lower() == "event":
        raw_r2 = raw_evt
        persist_r2 = persist_evt
    else:
        raw_r2 = raw_cut
        persist_r2 = persist_cut

    margin = float(getattr(cfg, "best_persist_margin", 0.0))
    penalty_w = float(getattr(cfg, "best_persist_penalty", 0.0))
    gap = (persist_r2 - raw_r2) - margin
    penalty = penalty_w * max(0.0, gap)

    bonus_w = float(getattr(cfg, "best_event_weight", 0.0))
    bonus = bonus_w * (raw_evt - persist_evt)

    return (raw_r2 - penalty + bonus, raw_r2, persist_r2)

def calibrate_dy_affine_on_target(model: GRUStepModel,
                                  target_runs: List[RunData],
                                  scaler: FeatureScaler,
                                  resid_cfg: ResidualFeedbackConfig,
                                  cfg: TrainConfig,
                                  max_points: int = 200000,
                                  min_keep_delta: float = 1e-6) -> None:
    """Analytic 1-shot calibration for dy_hat_norm -> dy_true_norm on target run(s).

    - Fits gain,bias in dy_norm space so C# can consume ONNX without extra changes.
    - Safety: compare target score before/after; keep only if it improves.
    - Calibration is computed from the *base* model output (dy_gain=1, dy_bias=0) to avoid compounding.
    """
    import numpy as np
    import torch

    if not target_runs:
        return

    model.eval()
    device = next(model.parameters()).device

    # Preserve original (in case user loaded a checkpoint that already had non-identity calib)
    g_prev = float(model.dy_gain.detach().cpu().numpy().reshape(-1)[0])
    b_prev = float(model.dy_bias.detach().cpu().numpy().reshape(-1)[0])

    # Use base output as reference for calibration
    with torch.no_grad():
        model.dy_gain.fill_(1.0)
        model.dy_bias.fill_(0.0)

    # Raw/persist R2 before calibration (base)
    try:
        pre_res = eval_torch(model, target_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
        pre_raw_r2 = float(pre_res.get("torch", {}).get("R2", float("nan")))
        pre_persist_r2 = float(pre_res.get("persist_baseline", {}).get("R2", float("nan")))
    except Exception:
        pre_raw_r2, pre_persist_r2 = float("nan"), float("nan")

    xs: List[float] = []
    ys: List[float] = []

    dy_mean = float(scaler.dy_mean)
    dy_std = float(scaler.dy_std)

    with torch.no_grad():
        for run in target_runs:
            N = int(run.y_raw.shape[0])
            if N < 2:
                continue

            cut0 = _first_cut_index(run.train_mask)
            if cut0 is None or cut0 >= N - 1:
                continue

            mapped = run.mapped_t
            look   = torch.tensor(run.look, dtype=torch.float32, device=device)
            y_raw  = torch.tensor(run.y_raw, dtype=torch.float32, device=device)
            mask   = torch.tensor(run.train_mask, dtype=torch.bool, device=device)

            h = torch.zeros(cfg.num_layers, 1, cfg.hidden_size, device=device)
            resid = ResidualFeedbackState(resid_cfg)
            resid.reset()

            for t in range(int(cut0), N - 1):
                do_eval = bool(mask[t].item() and mask[t + 1].item())
                if not do_eval:
                    if bool(getattr(cfg, "reset_on_aircut", True)):
                        h.zero_()
                        resid.reset()
                    continue

                r_t = resid.compute_r()
                r_t_ten = torch.tensor([[r_t]], dtype=torch.float32, device=device)
                x_t = torch.cat([mapped[t:t+1, :], look[t:t+1, :], r_t_ten], dim=1)

                # base dy_hat_norm (alpha applied; dy_gain/bias are identity here)
                dy_hat_next_norm, _, h = model.forward_with_alpha(x_t, h)
                dy_hat_next_norm = torch.clamp(dy_hat_next_norm, -float(cfg.dy_clip_norm), float(cfg.dy_clip_norm))

                y_next_raw = y_raw[t+1:t+2, :]
                y_prev_raw = y_raw[t:t+1, :]
                dy_next_raw = (y_next_raw - y_prev_raw)
                dy_next_norm = (dy_next_raw - dy_mean) / dy_std

                # resid update (teacher-forced)
                dy_hat_next_raw = dy_hat_next_norm * dy_std + dy_mean
                y_hat_next_raw = y_prev_raw + dy_hat_next_raw
                e_next = float((y_next_raw - y_hat_next_raw).detach().cpu().numpy()[0, 0])
                resid.update_with_new_residual(e_next)

                xs.append(float(dy_hat_next_norm.item()))
                ys.append(float(dy_next_norm.item()))
                if len(xs) >= int(max_points):
                    break

            if len(xs) >= int(max_points):
                break

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 1000:
        # restore prior calib if any
        with torch.no_grad():
            model.dy_gain.fill_(g_prev)
            model.dy_bias.fill_(b_prev)
        return

    vx = float(np.var(x))
    if vx < 1e-8:
        with torch.no_grad():
            model.dy_gain.fill_(g_prev)
            model.dy_bias.fill_(b_prev)
        return

    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    gain = cov / vx
    bias = float(y.mean() - gain * x.mean())

    # 안정적으로 clamp (필요시 조절)
    gain = float(np.clip(gain, float(getattr(cfg, 'target_calib_gain_min', 0.01)), float(getattr(cfg, 'target_calib_gain_max', 2.0))))
    bias = float(np.clip(bias, -float(getattr(cfg, 'target_calib_bias_abs', 2.0)), float(getattr(cfg, 'target_calib_bias_abs', 2.0))))

    with torch.no_grad():
        model.dy_gain.fill_(gain)
        model.dy_bias.fill_(bias)

    # Raw/persist R2 after calibration
    try:
        post_res = eval_torch(model, target_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
        post_raw_r2 = float(post_res.get("torch", {}).get("R2", float("nan")))
        post_persist_r2 = float(post_res.get("persist_baseline", {}).get("R2", float("nan")))
    except Exception:
        post_raw_r2, post_persist_r2 = float("nan"), float("nan")

    # Keep criterion: improve target raw R2 (avoid score side-effects)
    if not (post_raw_r2 > pre_raw_r2 + float(min_keep_delta)):
        # revert to identity calibration (base) – this is the safest default for deployment
        with torch.no_grad():
            model.dy_gain.fill_(1.0)
            model.dy_bias.fill_(0.0)
        print(f"[TARGET CALIB] reverted  pre_raw_r2={pre_raw_r2:.6f} post_raw_r2={post_raw_r2:.6f}")
        print(f"              pre_rawR2={pre_raw_r2:.4f} post_rawR2={post_raw_r2:.4f}  (gain,bias candidate=({gain:.6f},{bias:.6f}))")
    else:
        print(f"[TARGET CALIB] kept      pre_raw_r2={pre_raw_r2:.6f} post_raw_r2={post_raw_r2:.6f}")
        print(f"              pre_rawR2={pre_raw_r2:.4f} post_rawR2={post_raw_r2:.4f}  dy_gain={gain:.6f} dy_bias={bias:.6f}")

def train_stateful_gru(
    runs: List[RunData],
    scaler: FeatureScaler,
    resid_cfg: ResidualFeedbackConfig,
    cfg: TrainConfig,
    val_runs: Optional[List[RunData]] = None,
    target_runs: Optional[List[RunData]] = None,
    entry_weight: float = 1.3,
    entry_threshold: float = 0.5
) -> GRUStepModel:
    """Train stateful GRU.

    v23 upgrades:
      - persist-margin warmup 적용
      - best checkpoint 기준을 'RMSE' 외에 'raw R2 vs persist R2'로 선택 가능
      - 2-phase fine-tune(기본 ON): 1차에서 안정화 → 2차에서 deadband 완화/낮은 LR로 미세조정
    """
    val_runs = val_runs or []
    target_runs = target_runs or []

    device = torch.device(cfg.device)

    A = runs[0].mapped.shape[1]
    B = runs[0].look.shape[1]
    input_size = A + B + 1  # + residual feedback r_t

    model = GRUStepModel(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        head_type=cfg.head_type,
        head_dropout=cfg.head_dropout,
        bottleneck_dim=cfg.bottleneck_dim,
        use_alpha_gate=cfg.use_alpha_gate
    ).to(device)

    # v27: keep dy_gain/dy_bias fixed during training; only analytic calibration updates them
    if hasattr(model, "dy_gain"):
        model.dy_gain.requires_grad_(False)
    if hasattr(model, "dy_bias"):
        model.dy_bias.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # loss
    if cfg.loss_type == "mse":
        loss_vec = lambda yhat, y: (yhat - y) ** 2
    else:
        def loss_vec(yhat, y, delta: float = 1.0):
            abs_d = torch.abs(yhat - y)
            quad = 0.5 * (yhat - y) ** 2
            lin = delta * abs_d - 0.5 * delta
            return torch.where(abs_d <= delta, quad, lin)

    # entry_flag is the second-to-last column in sim.feats (before dMRR).
    sim_dim = runs[0].look.shape[1]  # same as sim_cur dim
    onm_dim = runs[0].mapped.shape[1] - sim_dim - 1  # subtract sim_cur and actLoad
    entry_idx_in_mapped = int(onm_dim + (sim_dim - 2))
    mrr_idx_in_mapped = int(onm_dim + 3)  # sim_cur MRR index (OrgF,OrgS,Depth,MRR,...)


    dy_mean = float(scaler.dy_mean)
    dy_std = float(scaler.dy_std)

    def _score_from_val(val_res: Dict) -> Tuple[float, float, float]:
        """Returns (score, raw_r2, persist_r2).

        - raw_r2 / persist_r2: chosen by cfg.best_focus ("cut" or "event")
        - score: raw_r2 minus penalty if raw is worse than persist by cfg.best_persist_margin,
                 plus a small event-gain bonus to avoid missing good event-tracking checkpoints.
        """
        # cut (default) metrics: mask[t] & mask[t+1]
        raw_cut = float(val_res.get("torch", {}).get("R2", float("-inf")))
        persist_cut = float(val_res.get("persist_baseline", {}).get("R2", float("-inf")))

        # event-only metrics: |dy_true_norm| >= threshold
        raw_evt = float(val_res.get("torch_event", {}).get("R2", raw_cut))
        persist_evt = float(val_res.get("persist_event", {}).get("R2", persist_cut))

        if str(cfg.best_focus).lower() == "event":
            raw_r2 = raw_evt
            persist_r2 = persist_evt
        else:
            raw_r2 = raw_cut
            persist_r2 = persist_cut

        # Penalize only when raw is worse than persist by more than margin
        gap = (persist_r2 - raw_r2) - float(cfg.best_persist_margin)
        penalty = float(cfg.best_persist_penalty) * max(0.0, gap)

        # Bonus: prefer checkpoints that beat persistence on events (tie-break / stability)
        bonus = float(cfg.best_event_weight) * (raw_evt - persist_evt)

        return (raw_r2 - penalty + bonus, raw_r2, persist_r2)

    best_val_rmse = float("inf")
    best_score = float("-inf")
    best_state = None
    no_improve = 0
    best_epoch = -1
    best_stage = "train"

    def _maybe_update_best(global_ep: int, stage: str, val_rmse: float, val_score: float, improved: bool):
        nonlocal best_val_rmse, best_score, best_state, no_improve, best_epoch, best_stage
        if improved or best_state is None:
            best_val_rmse = min(best_val_rmse, float(val_rmse))
            best_score = max(best_score, float(val_score))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(global_ep)
            best_stage = stage
            no_improve = 0
        else:
            no_improve += 1

    def _run_stage(
        stage: str,
        max_epochs: int,
        optimizer: torch.optim.Optimizer,
        start_ep: int,
        dy_deadband_raw_eff: float,
        pm_lambda_scale: float,
        patience: int,
        eval_every: int,
        dy_weight_k_scale: float = 1.0,
        dy_clip_norm_scale: float = 1.0,
        stable_lambda_scale: float = 1.0,
        aircut_lambda_scale: float = 1.0,
        stage_runs: Optional[List[RunData]] = None
    ) -> int:
        """Run training epochs. Returns next start_ep."""
        nonlocal no_improve

        for ep in range(max_epochs):
            global_ep = start_ep + ep

            # warmups
            warm = min(1.0, float(global_ep + 1) / float(max(1, int(cfg.alpha_warmup_epochs))))
            alpha_l1_eff = float(cfg.alpha_l1) * warm
            dy_l2_eff = float(cfg.dy_l2) * warm

            # persist margin warmup (separate)
            pm_base = float(cfg.persist_margin_lambda) * float(pm_lambda_scale)
            pm_warm = int(cfg.persist_margin_warmup_epochs)
            if pm_base > 0 and pm_warm > 0:
                pm_eff = pm_base * min(1.0, float(global_ep + 1) / float(pm_warm))
            else:
                pm_eff = pm_base

            # stage-wise effective coefficients
            dy_clip_eff = float(cfg.dy_clip_norm) * float(dy_clip_norm_scale)
            dy_weight_k_eff = float(cfg.dy_weight_k) * float(dy_weight_k_scale)
            stable_lam_eff = float(cfg.stable_lambda) * float(stable_lambda_scale)
            aircut_lam_eff = float(cfg.aircut_lambda) * float(aircut_lambda_scale)
            dy_weight_pow_eff = float(cfg.dy_weight_pow)

            model.train()
            ep_loss = 0.0
            ep_count = 0

            for run in runs:
                ensure_run_tensors(run, device)
                mapped = run.mapped_t
                mapped_raw = run.mapped_raw_t
                look = run.look_t
                y_raw = run.y_raw_t
                mask = run.mask_t

                if mapped is None or mapped_raw is None or look is None or y_raw is None or mask is None:
                    continue

                N = mapped.shape[0]
                if N < 2:
                    continue

                cut0 = _first_cut_index(run.train_mask)
                if cut0 is None or cut0 >= N - 1:
                    continue

                h = torch.zeros(cfg.num_layers, 1, cfg.hidden_size, device=device)
                resid = ResidualFeedbackState(resid_cfg)
                resid.reset()

                optimizer.zero_grad(set_to_none=True)
                chunk_loss = 0.0
                chunk_len = 0

                aircut_streak = 0

                # speed: sample a random contiguous window each epoch (keeps temporal structure)
                start_t = int(cut0)
                end_t = int(N - 1)
                ratio = float(getattr(cfg, "train_sample_ratio", 1.0))
                if ratio < 0.999:
                    span = max(2, end_t - start_t)
                    win = int(max(2, round(span * max(0.05, min(1.0, ratio)))))
                    if win < span:
                        # random window inside [start_t, end_t)
                        w0 = int(np.random.randint(start_t, end_t - win + 1))
                        start_t = w0
                        end_t = w0 + win
                        # reset state at window start
                        h.zero_()
                        resid.reset()
                for t in range(start_t, end_t):

                    do_train = bool(mask[t].item() and mask[t + 1].item())

                    # v27: runtime parity — skip non-cut steps, optional streak-based reset
                    if not do_train:
                        aircut_streak += 1
                        if bool(getattr(cfg, "reset_on_aircut", True)) and aircut_streak >= int(getattr(cfg, "aircut_reset_min_steps", 1)):
                            h.zero_()
                            resid.reset()
                            aircut_streak = 0
                        else:
                            h = h.detach()
                        continue
                    else:
                        aircut_streak = 0

                    r_t = resid.compute_r()  # raw
                    r_t_ten = torch.tensor([[r_t]], dtype=torch.float32, device=device)

                    x_t = torch.cat([mapped[t:t+1, :], look[t:t+1, :], r_t_ten], dim=1)

                    # dy_used = alpha * dy_base  (alpha in [0,1])
                    dy_hat_next_norm, alpha, h = model.forward_with_alpha(x_t, h)
                    dy_hat_next_norm = torch.clamp(dy_hat_next_norm, -dy_clip_eff, dy_clip_eff)

                    # target Δy in normalized space
                    y_next_raw = y_raw[t+1:t+2, :]
                    y_prev_raw = y_raw[t:t+1, :]
                    dy_next_raw = (y_next_raw - y_prev_raw)

                    # deadband small dy to 0 to avoid chasing noise
                    if float(dy_deadband_raw_eff) > 0:
                        db = float(dy_deadband_raw_eff)
                        dy_next_raw = torch.where(torch.abs(dy_next_raw) < db, torch.zeros_like(dy_next_raw), dy_next_raw)

                    dy_next_norm = (dy_next_raw - dy_mean) / dy_std

                    # residual update in raw space (teacher-forced with true y_next_raw)
                    dy_hat_next_raw = dy_hat_next_norm * dy_std + dy_mean
                    y_hat_next_raw = y_prev_raw + dy_hat_next_raw
                    e_next = float((y_next_raw - y_hat_next_raw).detach().cpu().numpy()[0, 0])
                    resid.update_with_new_residual(e_next)

                    # entry weighting (keeps original behavior)
                    entry_val = float(mapped[t, entry_idx_in_mapped].detach().cpu().numpy())
                    w_entry = entry_weight if entry_val >= entry_threshold else 1.0
                    w_entry_ten = torch.tensor([[w_entry]], dtype=torch.float32, device=device)

                    # emphasize change events (|dy|)
                    dy_abs = torch.abs(dy_next_norm)
                    ratio = torch.clamp(dy_abs / max(float(cfg.dy_weight_clip), 1e-12), 0.0, 1.0)
                    w_dy = 1.0 + dy_weight_k_eff * (ratio ** dy_weight_pow_eff)

                    base = loss_vec(dy_hat_next_norm[:, 0], dy_next_norm[:, 0], float(cfg.huber_delta)).unsqueeze(1)
                    loss = (base * w_entry_ten * w_dy).mean()

                    # discourage being worse than persist (dy_persist=0) with warmup-scaled lambda
                    if float(pm_eff) > 0:
                        err_model = (dy_hat_next_norm - dy_next_norm)
                        sse_model = (err_model ** 2)
                        sse_persist = (dy_next_norm ** 2)
                        mrg = float(cfg.persist_margin)
                        loss_margin = torch.relu(sse_model - sse_persist + mrg).mean()
                        loss = loss + float(pm_eff) * loss_margin

                    # alpha supervision
                    if cfg.use_alpha_gate and float(cfg.alpha_sup) > 0:
                        a_scale = max(float(cfg.alpha_target_scale), 1e-12)
                        alpha_target = torch.clamp(torch.abs(dy_next_norm) / a_scale, 0.0, 1.0).detach()
                        loss = loss + float(cfg.alpha_sup) * ((alpha - alpha_target) ** 2).mean()

                    # stable/air-cut stabilizers (help raw become persist-grade while preserving event response)
                    if stable_lam_eff > 0:
                        stable_mask = (torch.abs(dy_next_norm) < float(cfg.stable_dy_thresh_norm)).float()
                        loss = loss + stable_lam_eff * (stable_mask * (dy_hat_next_norm ** 2)).mean()

                    if aircut_lam_eff > 0:
                        mrr_raw_t = mapped_raw[t:t+1, mrr_idx_in_mapped:mrr_idx_in_mapped+1]
                        air_mask = (mrr_raw_t <= float(cfg.aircut_mrr_eps)).float()
                        loss = loss + aircut_lam_eff * (air_mask * (dy_hat_next_norm ** 2)).mean()

                    # regularization
                    loss = loss + dy_l2_eff * (dy_hat_next_norm ** 2).mean()
                    if cfg.use_alpha_gate:
                        loss = loss + alpha_l1_eff * alpha.mean()

                    chunk_loss = chunk_loss + loss
                    chunk_len += 1

                    if chunk_len >= cfg.tbptt_steps:
                        chunk_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        h = h.detach()

                        ep_loss += float(chunk_loss.detach().cpu().numpy())
                        ep_count += chunk_len
                        chunk_loss = 0.0
                        chunk_len = 0

                if chunk_len > 0:
                    chunk_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ep_loss += float(chunk_loss.detach().cpu().numpy())
                    ep_count += chunk_len

            train_avg = ep_loss / max(1, ep_count)

            # validation / early stopping (val + optional target)
            if cfg.early_stop and ((global_ep + 1) % max(1, int(eval_every)) == 0) and (val_runs or target_runs):
                val_rmse = float("inf")
                val_score = float("-inf")
                raw_r2 = float("nan")
                persist_r2 = float("nan")

                if val_runs:
                    val_res = eval_torch(model, val_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
                    val_rmse = float(val_res.get("torch", {}).get("RMSE", float("inf")))
                    val_score, raw_r2, persist_r2 = _score_from_val(val_res)

                tgt_score = float("-inf")
                tgt_raw_r2 = float("nan")
                tgt_persist_r2 = float("nan")
                if target_runs:
                    tgt_res = eval_torch(model, target_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
                    tgt_score, tgt_raw_r2, tgt_persist_r2 = _score_from_val(tgt_res)

                bm = str(cfg.best_metric).lower()

                # choose best-score source
                if bm == "val_rmse":
                    if not val_runs:
                        score_for_best = val_score
                        improved = False
                    else:
                        score_for_best = val_score
                        improved = (best_val_rmse - val_rmse) > float(cfg.es_min_delta)
                else:
                    if bm == "target_r2_vs_persist":
                        score_for_best = tgt_score
                    elif bm == "mix_r2":
                        w = float(getattr(cfg, "target_score_weight", 0.70))
                        if (not val_runs) and target_runs:
                            score_for_best = tgt_score
                        elif val_runs and (not target_runs):
                            score_for_best = val_score
                        else:
                            score_for_best = (1.0 - w) * val_score + w * tgt_score
                    else:
                        score_for_best = val_score

                    improved = (score_for_best - best_score) > float(cfg.es_min_delta_score)

                _maybe_update_best(global_ep, stage, val_rmse, score_for_best, improved)

                msg = f"[{stage}] epoch {global_ep+1}  avg_loss={train_avg:.6f}  "
                if val_runs:
                    msg += f"val_RMSE={val_rmse:.6f}  val_raw_R2={raw_r2:.4f}  val_persist_R2={persist_r2:.4f}  "
                if target_runs:
                    msg += f"tgt_raw_R2={tgt_raw_r2:.4f}  tgt_persist_R2={tgt_persist_r2:.4f}  tgt_score={tgt_score:.4f}  "
                msg += f"score={score_for_best:.4f}  best_score={best_score:.4f}  bad={no_improve}/{patience}  metric={bm}"
                print(msg)

                if no_improve >= int(patience):
                    print(f"[EARLY STOP:{stage}] stop at epoch {global_ep+1} (best epoch {best_epoch+1}@{best_stage})")
                    return global_ep + 1
            else:
                print(f"[{stage}] epoch {global_ep+1}  avg_loss={train_avg:.6f}")


        return start_ep + max_epochs

    # ===== Stage 1 =====
    cur_ep = 0
    cur_ep = _run_stage(
        stage="train",
        max_epochs=int(cfg.epochs),
        optimizer=opt,
        start_ep=cur_ep,
        dy_deadband_raw_eff=float(cfg.dy_deadband_raw),
        pm_lambda_scale=1.0,
        patience=int(cfg.es_patience),
        eval_every=int(cfg.es_eval_every)
    )

    # load best from stage1
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== Stage 2 (fine-tune) =====
    if bool(cfg.finetune) and (val_runs or target_runs) and int(cfg.ft_epochs) > 0:
        print("\n[FINETUNE] starting fine-tune phase...")
        no_improve = 0  # reset patience counter

        ft_lr = float(cfg.lr) * float(cfg.ft_lr_scale)
        ft_wd = float(cfg.weight_decay) * float(cfg.ft_weight_decay_scale)
        opt2 = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=ft_wd)

        dy_deadband_ft = float(cfg.dy_deadband_raw) * float(cfg.ft_dy_deadband_scale)
        cur_ep = _run_stage(
            stage="ft",
            max_epochs=int(cfg.ft_epochs),
            optimizer=opt2,
            start_ep=cur_ep,
            dy_deadband_raw_eff=dy_deadband_ft,
            pm_lambda_scale=float(cfg.ft_persist_margin_lambda_scale),
            patience=int(cfg.ft_patience),
            eval_every=int(cfg.ft_eval_every),
            dy_weight_k_scale=float(cfg.ft_dy_weight_k_scale),
            dy_clip_norm_scale=float(cfg.ft_dy_clip_norm_scale),
            stable_lambda_scale=float(cfg.ft_stable_lambda_scale),
            aircut_lambda_scale=float(cfg.ft_aircut_lambda_scale)
        )

        if best_state is not None:
            model.load_state_dict(best_state)


    
    # ===== Stage 3 (target fine-tune) =====
    # 목적: 특정 타겟 런(O81 등)에서 raw를 더 끌어올리기 위한 짧은 적응(finetune)
    if target_runs and bool(getattr(cfg, "target_finetune", False)) and int(getattr(cfg, "target_ft_epochs", 0)) > 0:
        print()
        print("[TARGET-FT] starting target fine-tune phase...")
        no_improve = 0  # reset patience counter

        # start from current best
        if best_state is not None:
            model.load_state_dict(best_state)

        freeze_gru = bool(getattr(cfg, "target_freeze_gru", True))
        if freeze_gru:
            for p in model.gru.parameters():
                p.requires_grad_(False)

        tgt_lr = float(cfg.lr) * float(getattr(cfg, "target_ft_lr_scale", 0.05))
        tgt_wd = float(cfg.weight_decay) * float(getattr(cfg, "target_ft_weight_decay_scale", 0.2))

        params = [p for p in model.parameters() if p.requires_grad]
        opt3 = torch.optim.AdamW(params, lr=tgt_lr, weight_decay=tgt_wd)

        # NOTE: 이 stage는 실제로 target_runs만 학습(runs override)하며,
        #       best_metric을 target_r2_vs_persist로 두면 타겟 raw 기준으로 checkpoint가 선택됨.
        cur_ep = _run_stage(
            stage="tgtft",
            max_epochs=int(getattr(cfg, "target_ft_epochs", 0)),
            optimizer=opt3,
            start_ep=cur_ep,
            dy_deadband_raw_eff=float(cfg.dy_deadband_raw),
            pm_lambda_scale=1.0,
            patience=int(getattr(cfg, "ft_patience", 20)),
            eval_every=int(getattr(cfg, "ft_eval_every", 1)),
            stage_runs=target_runs
        )

        # restore GRU trainability for any later stages (safety)
        if freeze_gru:
            for p in model.gru.parameters():
                p.requires_grad_(True)

        # keep best weights after target-FT
        if best_state is not None:
            model.load_state_dict(best_state)


    # load best weights before returning / exporting
    if best_state is not None:
        model.load_state_dict(best_state)

    # v26: target calibration (analytic gain/bias in dy_norm space)
    # v27: target calibration (analytic gain/bias in dy_norm space) — keep only if it improves target score
    if target_runs and bool(getattr(cfg, "target_calib", True)):
        try:
            pre_res = eval_torch(model, target_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
            pre_score, pre_raw_r2, pre_persist_r2 = score_from_val(pre_res, cfg)
        except Exception:
            pre_raw_r2, pre_raw_r2, pre_persist_r2 = float("-inf"), float("nan"), float("nan")

        old_gain = float(model.dy_gain.detach().cpu().numpy()[0]) if hasattr(model, "dy_gain") else 1.0
        old_bias = float(model.dy_bias.detach().cpu().numpy()[0]) if hasattr(model, "dy_bias") else 0.0

        calibrate_dy_affine_on_target(model, target_runs, scaler, resid_cfg, cfg)

        try:
            post_res = eval_torch(model, target_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
            post_score, post_raw_r2, post_persist_r2 = score_from_val(post_res, cfg)
        except Exception:
            post_raw_r2, post_raw_r2, post_persist_r2 = float("-inf"), float("nan"), float("nan")

        if not (post_raw_r2 > pre_raw_r2 + 1e-6):
            with torch.no_grad():
                if hasattr(model, "dy_gain"):
                    model.dy_gain.copy_(torch.tensor([old_gain], dtype=torch.float32, device=next(model.parameters()).device))
                if hasattr(model, "dy_bias"):
                    model.dy_bias.copy_(torch.tensor([old_bias], dtype=torch.float32, device=next(model.parameters()).device))
            print(f"[TARGET CALIB-KEEP] reverted (pre_raw_r2={pre_raw_r2:.4f} post_raw_r2={post_raw_r2:.4f})")
        else:
            print(f"[TARGET CALIB-KEEP] kept (pre_raw_r2={pre_raw_r2:.4f} post_raw_r2={post_raw_r2:.4f})")


    # v3: validation calibration (analytic dy_gain/dy_bias in dy_norm space) when no target is provided
    # - This is NOT a smoothing post-process. It adjusts dy_gain/dy_bias parameters baked into ONNX.
    # - Keep the calibration only if it improves raw R2 on validation (according to score_from_val focus).
    if (not target_runs) and val_runs and bool(getattr(cfg, "val_calib", True)):
        try:
            pre_res = eval_torch(model, val_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
            _, pre_raw_r2, _ = score_from_val(pre_res, cfg)
        except Exception:
            pre_raw_r2 = float("-inf")

        old_gain = float(model.dy_gain.detach().cpu().numpy().reshape(-1)[0]) if hasattr(model, "dy_gain") else 1.0
        old_bias = float(model.dy_bias.detach().cpu().numpy().reshape(-1)[0]) if hasattr(model, "dy_bias") else 0.0

        calibrate_dy_affine_on_target(model, val_runs, scaler, resid_cfg, cfg)

        try:
            post_res = eval_torch(model, val_runs, scaler, resid_cfg, cfg, dump_csv=None, dump_max=0)
            _, post_raw_r2, _ = score_from_val(post_res, cfg)
        except Exception:
            post_raw_r2 = float("-inf")

        if not (post_raw_r2 > pre_raw_r2 + 1e-6):
            with torch.no_grad():
                dev = next(model.parameters()).device
                if hasattr(model, "dy_gain"):
                    model.dy_gain.copy_(torch.tensor([old_gain], dtype=torch.float32, device=dev))
                if hasattr(model, "dy_bias"):
                    model.dy_bias.copy_(torch.tensor([old_bias], dtype=torch.float32, device=dev))
            print(f"[VAL CALIB-KEEP] reverted (pre_raw_r2={pre_raw_r2:.4f} post_raw_r2={post_raw_r2:.4f})")
        else:
            print(f"[VAL CALIB-KEEP] kept (pre_raw_r2={pre_raw_r2:.4f} post_raw_r2={post_raw_r2:.4f})")

    print(f"\n[BEST] epoch={best_epoch+1} stage={best_stage}  best_RMSE={best_val_rmse:.6f}  best_score={best_score:.4f}")
    return model


# =========================
# 10) ONNX Export
# =========================

def export_onnx(model: GRUStepModel, onnx_path: str, input_size: int, cfg: TrainConfig):
    model.eval()
    dummy_x = torch.randn(1, input_size, dtype=torch.float32)
    dummy_h = torch.zeros(cfg.num_layers, 1, cfg.hidden_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x, dummy_h),
        onnx_path,
        input_names=["x", "h_in"],
        output_names=["y", "h_out"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[ONNX] exported: {onnx_path}")


# =========================
# 11) Pairing sim/act files
# =========================

def _norm_num_str(s: str) -> str:
    """Normalize numeric string so that 0.50 == 0.5 == 0.500 etc."""
    s = str(s).strip()
    if s == "":
        return s
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _extract_feed_depth(path: str, kind: str) -> Optional[Tuple[str, str]]:
    """Extract (F, D) from filename.

    Expected patterns (case-insensitive):
      - Sim: Sim_F1200_D0.5.csv
      - Act: Act_F1200_D0.5_*.csv

    Returns:
      ("1200", "0.5") or None when not matched.
    """
    base = os.path.basename(path)
    m = re.search(rf"{kind}_F(\d+)_D([0-9]+(?:\.[0-9]+)?)", base, flags=re.IGNORECASE)
    if not m:
        return None
    f_str = m.group(1)
    d_str = _norm_num_str(m.group(2))
    return (f_str, d_str)


def find_pairs_by_feed_depth(paths_cfg: PathsConfig) -> List[Tuple[str, str, str]]:
    """Pair Sim/Act files by 동일 Feed(F) + Depth(D) in filename.

    - sim_glob: learning_data/Sim/*.csv  (Sim_F*_D*.csv)
    - act_glob: learning_data/Act/*.csv  (Act_F*_D*_*.csv)

    Returns list of (key, sim_path, act_path)
      key example: "F1200_D0.5"
    """
    sim_files = sorted(glob.glob(paths_cfg.sim_glob))
    act_files = sorted(glob.glob(paths_cfg.act_glob))

    sim_map: Dict[Tuple[str, str], str] = {}
    for sp in sim_files:
        key = _extract_feed_depth(sp, "Sim")
        if not key:
            continue
        # if duplicates exist, keep first one deterministically
        sim_map.setdefault(key, sp)

    pairs: List[Tuple[str, str, str]] = []
    for ap in act_files:
        key = _extract_feed_depth(ap, "Act")
        if not key:
            continue
        sp = sim_map.get(key)
        if not sp:
            continue
        key_str = f"F{key[0]}_D{key[1]}"
        pairs.append((key_str, sp, ap))

    pairs.sort(key=lambda x: (x[0], x[2]))
    return pairs

# =========================
# 12) Eval + dump (raw unit metrics)
# =========================

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, pred_times_ms: Optional[List[float]] = None):
    y_true = y_true.reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    if y_true.size == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE (%)": np.nan,
            "NRMSE (%)": np.nan,
            "R2": np.nan,
            "R²": np.nan,
            "Peak error (%)": np.nan,
            "Avg pred. time (ms)": np.nan,
            "P95 pred. time (ms)": np.nan,
        }

    err = y_true - y_pred
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))

    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - (np.sum(err ** 2) / denom)) if denom > 1e-12 else np.nan

    ape_denom = np.maximum(np.abs(y_true), 1e-12)
    mape = float(np.mean(abs_err / ape_denom) * 100.0)

    y_range = float(np.max(y_true) - np.min(y_true))
    nrmse = float((rmse / y_range) * 100.0) if y_range > 1e-12 else np.nan

    peak_true = float(np.max(y_true)) if y_true.size > 0 else np.nan
    peak_pred = float(np.max(y_pred)) if y_pred.size > 0 else np.nan
    peak_err = float(abs(peak_true - peak_pred) / max(abs(peak_true), 1e-12) * 100.0) if np.isfinite(peak_true) else np.nan

    avg_pred_ms = np.nan
    p95_pred_ms = np.nan
    if pred_times_ms is not None:
        pred_arr = np.asarray(pred_times_ms, dtype=np.float64).reshape(-1)
        pred_arr = pred_arr[np.isfinite(pred_arr)]
        if pred_arr.size > 0:
            avg_pred_ms = float(np.mean(pred_arr))
            p95_pred_ms = float(np.percentile(pred_arr, 95.0))

    return {
        "N": int(y_true.size),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "NRMSE (%)": nrmse,
        "R2": r2,
        "R²": r2,
        "Peak error (%)": peak_err,
        "Avg pred. time (ms)": avg_pred_ms,
        "P95 pred. time (ms)": p95_pred_ms,
    }


def eval_torch(model: GRUStepModel, runs: List[RunData],
              scaler: FeatureScaler,
              resid_cfg: ResidualFeedbackConfig, cfg: TrainConfig,
              dump_csv: Optional[str] = None, dump_max: int = 20000):
    device = torch.device(cfg.device)
    model.eval()

    # Use the provided runs list (avoid relying on an outer-scope variable).
    A = runs[0].mapped.shape[1]
    B = runs[0].look.shape[1]
    input_size = A + B + 1

    y_mean = float(scaler.y_mean)
    dy_mean = float(scaler.dy_mean)
    dy_std = float(scaler.dy_std)
    y_std = float(scaler.y_std)

    ys_true, ys_pred = [], []
    ys_pred_mean, ys_pred_persist = [], []
    ys_dy_true_norm = []
    pred_times_ms = []

    dump_rows = []
    dumped = 0

    with torch.no_grad():
        for run_i, run in enumerate(runs):
            ensure_run_tensors(run, device)
            if run.mapped_t is None or run.look_t is None or run.y_raw_t is None or run.mask_t is None:
                continue
            N = run.mapped.shape[0]
            if N < 2:
                continue

            cut0 = _first_cut_index(run.train_mask)
            if cut0 is None or cut0 >= N - 1:
                continue

            mapped = run.mapped_t
            look   = run.look_t
            y_raw  = run.y_raw_t
            y_norm = torch.from_numpy(run.y.astype(np.float32)).to(device, non_blocking=True) if run.y is not None else None
            mask   = run.mask_t
            if y_norm is None:
                continue

            h = torch.zeros(cfg.num_layers, 1, cfg.hidden_size, device=device)
            resid = ResidualFeedbackState(resid_cfg)
            resid.reset()

            # mean baseline (raw)
            eval_y_next = []
            aircut_streak = 0

            for t in range(cut0, N - 1):
                if bool(mask[t].item() and mask[t + 1].item()):
                    eval_y_next.append(float(y_raw[t + 1, 0].item()))
            mean_base = float(np.mean(eval_y_next)) if len(eval_y_next) else 0.0

            for t in range(cut0, N - 1):
                do_eval = bool(mask[t].item() and mask[t + 1].item())

                # v27: runtime parity — skip non-cut steps, optional streak-based reset
                if not do_eval:
                    aircut_streak += 1
                    if bool(getattr(cfg, "reset_on_aircut", True)) and aircut_streak >= int(getattr(cfg, "aircut_reset_min_steps", 1)):
                        h.zero_()
                        resid.reset()
                        aircut_streak = 0
                    continue
                else:
                    aircut_streak = 0

                r_t = resid.compute_r()
                r_t_ten = torch.tensor([[r_t]], dtype=torch.float32, device=device)

                x_t = torch.cat([mapped[t:t+1, :], look[t:t+1, :], r_t_ten], dim=1)
                _t0 = time.perf_counter()
                dy_hat_next_norm, h = model(x_t, h)
                pred_times_ms.append((time.perf_counter() - _t0) * 1000.0)

                y_next_raw = y_raw[t+1:t+2, :]
                y_prev_raw = y_raw[t:t+1, :]
                # y_next_norm = y_norm[t+1:t+2, :]  # (not used for Δy loss)

                dy_hat_next_raw = dy_hat_next_norm * dy_std + dy_mean
                y_hat_next_raw = y_prev_raw + dy_hat_next_raw

                e_next = float((y_next_raw - y_hat_next_raw).detach().cpu().numpy()[0, 0])
                resid.update_with_new_residual(e_next)

                yt_raw = float(y_raw[t, 0].item())
                ytrue = float(y_next_raw[0, 0].item())
                ypred = float(y_hat_next_raw[0, 0].item())
                dy_true_raw = float((y_next_raw - y_prev_raw)[0, 0].item())
                dy_true_norm = float((dy_true_raw - dy_mean) / max(dy_std, 1e-12))

                ys_true.append(ytrue)
                ys_pred.append(ypred)
                ys_dy_true_norm.append(dy_true_norm)
                ys_pred_mean.append(mean_base)
                ys_pred_persist.append(yt_raw)

                if dump_csv and dumped < dump_max:
                    dump_rows.append([
                        run_i, t,
                        ytrue, ypred, (ytrue - ypred),
                        float(r_t),
                        mean_base,
                        yt_raw,
                        float(pred_times_ms[-1]) if pred_times_ms else np.nan
                    ] + x_t.detach().cpu().numpy().reshape(-1).tolist())
                    dumped += 1

    y_true = np.array(ys_true, dtype=np.float32)
    y_pred = np.array(ys_pred, dtype=np.float32)

    out = {
        "torch": _metrics(y_true, y_pred, pred_times_ms=pred_times_ms),
        "mean_baseline": _metrics(y_true, np.array(ys_pred_mean, dtype=np.float32)),
        "persist_baseline": _metrics(y_true, np.array(ys_pred_persist, dtype=np.float32)),
        "input_size": input_size
    }

    if dump_csv:
        header = ["run", "t", "y_true_next", "y_pred_next", "residual_next", "r_t",
                  "mean_base", "persist_base", "pred_time_ms"] + [f"x_{i}" for i in range(out["input_size"])]
        with open(dump_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(dump_rows)

    # event-only metrics (focus on large |dy| changes)
    try:
        dy_true_norm_arr = np.array(ys_dy_true_norm, dtype=np.float32)
        if dy_true_norm_arr.size > 0:
            ev_mask = (np.abs(dy_true_norm_arr) >= float(cfg.best_event_dy_thresh_norm))
            if np.any(ev_mask):
                out["torch_event"] = _metrics(y_true[ev_mask], y_pred[ev_mask])
                out["persist_event"] = _metrics(
                    y_true[ev_mask],
                    np.array(ys_pred_persist, dtype=np.float32)[ev_mask]
                )
    except Exception:
        pass


    return out


def eval_onnx(onnx_path: str, runs: List[RunData],
             scaler: FeatureScaler,
             resid_cfg: ResidualFeedbackConfig, cfg: TrainConfig,
             input_size: int,
             dump_csv: Optional[str] = None, dump_max: int = 20000):
    try:
        import onnxruntime as ort
    except Exception as e:
        print("[WARN] onnxruntime not installed, skip eval_onnx:", e)
        return {"onnx": {"N": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}}

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    x_name = "x" if "x" in in_names else in_names[0]
    h_name = "h_in" if "h_in" in in_names else in_names[-1]
    y_name = "y" if "y" in out_names else out_names[0]
    ho_name = "h_out" if "h_out" in out_names else out_names[-1]

    y_mean = float(scaler.y_mean)
    dy_mean = float(scaler.dy_mean)
    dy_std = float(scaler.dy_std)
    y_std = float(scaler.y_std)

    ys_true, ys_pred = [], []
    ys_pred_mean, ys_pred_persist = [], []
    ys_dy_true_norm = []
    pred_times_ms = []

    dump_rows = []
    dumped = 0

    for run_i, run in enumerate(runs):
        N = run.mapped.shape[0]
        if N < 2:
            continue

        cut0 = _first_cut_index(run.train_mask)
        if cut0 is None or cut0 >= N - 1:
            continue

        mapped = run.mapped.astype(np.float32)
        look   = run.look.astype(np.float32)
        y_raw  = run.y_raw.astype(np.float32)
        mask   = run.train_mask.astype(np.bool_)

        h = np.zeros((cfg.num_layers, 1, cfg.hidden_size), dtype=np.float32)
        resid = ResidualFeedbackState(resid_cfg)
        resid.reset()
        aircut_streak = 0

        eval_y_next = []
        for t in range(cut0, N - 1):
            if bool(mask[t] and mask[t + 1]):
                eval_y_next.append(float(y_raw[t + 1, 0]))
        mean_base = float(np.mean(eval_y_next)) if len(eval_y_next) else 0.0

        for t in range(cut0, N - 1):
            do_eval = bool(mask[t] and mask[t + 1])

            # v27: runtime parity — skip non-cut steps, optional streak-based reset
            if not do_eval:
                aircut_streak += 1
                if bool(getattr(cfg, "reset_on_aircut", True)) and aircut_streak >= int(getattr(cfg, "aircut_reset_min_steps", 1)):
                    h[:] = 0.0
                    resid.reset()
                    aircut_streak = 0
                continue
            else:
                aircut_streak = 0

            r_t = float(resid.compute_r())
            x_vec = np.concatenate([mapped[t, :], look[t, :], np.array([r_t], dtype=np.float32)], axis=0)
            x_in = x_vec.reshape(1, input_size).astype(np.float32)

            _t0 = time.perf_counter()
            y_out, h_out = sess.run([y_name, ho_name], {x_name: x_in, h_name: h})
            pred_times_ms.append((time.perf_counter() - _t0) * 1000.0)
            dy_pred_next_norm = float(np.asarray(y_out).reshape(-1)[0])
            h = np.asarray(h_out).astype(np.float32)

            dy_pred_next_raw = dy_pred_next_norm * dy_std + dy_mean
            y_prev_raw = float(y_raw[t, 0])
            y_pred_next_raw = y_prev_raw + dy_pred_next_raw
            y_true_next_raw = float(y_raw[t + 1, 0])

            e_next = y_true_next_raw - y_pred_next_raw
            resid.update_with_new_residual(e_next)

            yt_raw = float(y_raw[t, 0])

            dy_true_raw = float(y_true_next_raw - y_prev_raw)
            dy_true_norm = float((dy_true_raw - dy_mean) / max(dy_std, 1e-12))

            ys_true.append(y_true_next_raw)
            ys_pred.append(y_pred_next_raw)
            ys_dy_true_norm.append(dy_true_norm)
            ys_pred_mean.append(mean_base)
            ys_pred_persist.append(yt_raw)

            if dump_csv and dumped < dump_max:
                dump_rows.append([
                    run_i, t,
                    y_true_next_raw, y_pred_next_raw, e_next,
                    r_t,
                    mean_base,
                    yt_raw,
                    float(pred_times_ms[-1]) if pred_times_ms else np.nan
                ] + x_vec.tolist())
                dumped += 1

    y_true = np.array(ys_true, dtype=np.float32)
    y_pred = np.array(ys_pred, dtype=np.float32)

    out = {
        "onnx": _metrics(y_true, y_pred, pred_times_ms=pred_times_ms),
        "mean_baseline": _metrics(y_true, np.array(ys_pred_mean, dtype=np.float32)),
        "persist_baseline": _metrics(y_true, np.array(ys_pred_persist, dtype=np.float32)),
        "input_size": input_size,
        "io": {"inputs": in_names, "outputs": out_names, "x": x_name, "h": h_name, "y": y_name, "h_out": ho_name}
    }

    if dump_csv:
        header = ["run", "t", "y_true_next", "y_pred_next", "residual_next", "r_t",
                  "mean_base", "persist_base", "pred_time_ms"] + [f"x_{i}" for i in range(out["input_size"])]
        with open(dump_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(dump_rows)

    # event-only metrics (focus on large |dy| changes)
    try:
        dy_true_norm_arr = np.array(ys_dy_true_norm, dtype=np.float32)
        if dy_true_norm_arr.size > 0:
            ev_mask = (np.abs(dy_true_norm_arr) >= float(cfg.best_event_dy_thresh_norm))
            if np.any(ev_mask):
                out["onnx_event"] = _metrics(y_true[ev_mask], y_pred[ev_mask])
                out["persist_event"] = _metrics(
                    y_true[ev_mask],
                    np.array(ys_pred_persist, dtype=np.float32)[ev_mask]
                )
    except Exception:
        pass


    return out


def compare_torch_onnx_firstK(torch_dump_csv: str, onnx_dump_csv: str, k: int = 2000):
    def read_rows(path, k):
        rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            for i, row in enumerate(r):
                if i >= k:
                    break
                rows.append(row)
        return header, rows

    ht, rt = read_rows(torch_dump_csv, k)
    ho, ro = read_rows(onnx_dump_csv, k)

    idx_y_pred = ht.index("y_pred_next")
    idx_x0 = ht.index("x_0")

    n = min(len(rt), len(ro))
    if n == 0:
        print("[COMPARE] no rows")
        return

    max_abs_pred = 0.0
    max_abs_x = 0.0

    for i in range(n):
        ypt = float(rt[i][idx_y_pred])
        ypo = float(ro[i][idx_y_pred])
        max_abs_pred = max(max_abs_pred, abs(ypt - ypo))

        for j in range(idx_x0, len(rt[i])):
            xt = float(rt[i][j])
            xo = float(ro[i][j])
            max_abs_x = max(max_abs_x, abs(xt - xo))

    print(f"[COMPARE] rows={n}, max|y_pred_torch-onnx|={max_abs_pred:.6g}, max|x_torch-onnx|={max_abs_x:.6g}")


# =========================
# 13) Main
# =========================


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    if s is None:
        return out
    for part in str(s).split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    if s is None:
        return out
    for part in str(s).split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train stateful GRU for cutting-load prediction (sim+on-machine fusion).")

    # ===== Train =====
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--tbptt-steps", type=int, default=160)
    p.add_argument("--train-sample-ratio", type=float, default=0.35, help="Train on random sub-window ratio per run per epoch (0<r<=1). Use 1.0 for full scan.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # ===== Early stopping / Validation =====
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Validation split ratio by runs (prefixes). 0 disables validation/early-stopping.")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping even if validation runs exist.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (number of eval checks without improvement).")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="Minimum RMSE improvement to reset patience.")
    p.add_argument("--val-every", type=int, default=5,
                   help="Evaluate validation every N epochs.")

    # ===== Best model selection (validation) =====
    p.add_argument("--best-metric", type=str, default="r2_vs_persist",
                   choices=["val_rmse", "r2_vs_persist", "target_r2_vs_persist", "mix_r2"],
                   help="Best checkpoint criterion. 'r2_vs_persist' targets raw-R2 while avoiding worse-than-persist behavior.")
    p.add_argument("--best-persist-penalty", type=float, default=2.0,
                   help="Penalty weight when raw R2 is worse than persist baseline R2.")
    p.add_argument("--best-persist-margin", type=float, default=0.0,
                   help="Allowed raw-vs-persist R2 gap before applying penalty. 0 means 'raw should not be worse than persist'.")
    p.add_argument("--best-focus", type=str, default="event", choices=["cut", "event"],
                   help="Which subset metric is used for best selection. 'event' focuses on large |dy| changes.")
    p.add_argument("--best-event-dy-thresh-norm", type=float, default=0.35,
                   help="Event threshold in normalized dy units (used when best-focus=event and for event bonus).")
    p.add_argument("--best-event-weight", type=float, default=0.25,
                   help="Small bonus weight using (R2_event_raw - R2_event_persist) to avoid missing good event-tracking checkpoints.")
    p.add_argument("--min-delta-score", type=float, default=1e-4,
                   help="Minimum score improvement to reset patience (for best-metric=r2_vs_persist).")


    # ===== Target(O81) optimizing (optional) =====
    # Allow multiple targets by repeating flags:
    #   --target-sim SIM1 --target-act ACT1 --target-sim SIM2 --target-act ACT2
    p.add_argument("--target-sim", action="append", default=[],
                   help="Target SIM csv path(s). Repeat flag to add multiple.")
    p.add_argument("--target-act", action="append", default=[],
                   help="Target ACT csv path(s). Repeat flag to add multiple.")
    p.add_argument("--scaler-fit-target", action="store_true",
                   help="Include target run(s) when fitting scaler.")

    p.add_argument("--target-score-weight", type=float, default=0.70,
                   help="Weight for target score when --best-metric=mix_r2 (0..1).")

    p.add_argument("--no-reset-on-aircut", action="store_true",
                   help="Disable runtime-parity reset on non-cut/aircut.")
    p.add_argument("--no-aircut-hold-persist", action="store_true",
                   help="Disable eval-only persist hold on aircut.")


    p.add_argument("--aircut-reset-min-steps", type=int, default=1,
                   help="Reset hidden/resid only after N consecutive non-cut steps (1 = immediate).")

    p.add_argument("--no-val-calib", action="store_true",
                   help="Disable validation calibration (dy_gain/dy_bias) when no target run is provided.")

    p.add_argument("--no-target-calib", action="store_true",
                   help="Disable target calibration stage.")
    p.add_argument("--target-calib-full", action="store_true",
                   help="If set, allow calibration to train more params (not only dy_gain/bias).")
    p.add_argument("--target-calib-epochs", type=int, default=120)
    p.add_argument("--target-calib-lr-scale", type=float, default=0.20)

    # Calibration clamps (dy_norm space)
    p.add_argument("--target-calib-gain-min", type=float, default=0.01)
    p.add_argument("--target-calib-gain-max", type=float, default=2.0)
    p.add_argument("--target-calib-bias-abs", type=float, default=2.0)

    p.add_argument("--no-target-finetune", action="store_true",
                   help="Disable target finetune stage after calibration.")
    p.add_argument("--target-ft-epochs", type=int, default=80)
    p.add_argument("--target-ft-lr-scale", type=float, default=0.05)
    p.add_argument("--no-target-freeze-gru", action="store_true",
                   help="If set, do NOT freeze GRU during target finetune.")
# ===== Finetune (2-phase) =====
    p.add_argument("--finetune", dest="finetune", action="store_true",
                   help="Enable fine-tune phase after the best checkpoint is found (default: enabled).")
    p.add_argument("--no-finetune", dest="finetune", action="store_false",
                   help="Disable fine-tune phase.")
    p.set_defaults(finetune=True)
    p.add_argument("--ft-epochs", type=int, default=160)
    p.add_argument("--ft-lr-scale", type=float, default=0.08, help="Fine-tune LR = lr * scale.")
    p.add_argument("--ft-weight-decay-scale", type=float, default=0.4, help="Fine-tune weight_decay = weight_decay * scale.")
    p.add_argument("--ft-dy-deadband-scale", type=float, default=0.20, help="Fine-tune dy_deadband_raw = dy_deadband_raw * scale.")
    p.add_argument("--ft-persist-margin-lambda-scale", type=float, default=0.7, help="Fine-tune persist_margin_lambda *= scale.")
    p.add_argument("--ft-dy-weight-k-scale", type=float, default=1.4, help="Fine-tune: dy_weight_k *= scale (event focus).")
    p.add_argument("--ft-dy-clip-norm-scale", type=float, default=1.0, help="Fine-tune: dy_clip_norm *= scale.")
    p.add_argument("--ft-stable-lambda-scale", type=float, default=0.6, help="Fine-tune: stable_lambda *= scale (reduce over-smoothing).")
    p.add_argument("--ft-aircut-lambda-scale", type=float, default=0.8, help="Fine-tune: aircut_lambda *= scale.")
    p.add_argument("--ft-patience", type=int, default=20)
    p.add_argument("--ft-val-every", type=int, default=5)


    # ===== Head =====
    p.add_argument("--head", type=str, default="residual",
                   choices=["baseline", "bottleneck", "residual", "linear"])
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--bottleneck-dim", type=int, default=96)

    # ===== Sweep mode (head 메뉴) =====
    p.add_argument("--sweep", action="store_true",
                   help="Run a small sweep menu over head configurations (recommended for quick comparison).")
    p.add_argument("--sweep-outdir", type=str, default="sweep_outputs",
                   help="Base output directory for sweep runs.")
    p.add_argument("--sweep-dropouts", type=str, default="0.05,0.1",
                   help="Comma-separated dropout list used for bottleneck/residual heads in sweep mode. e.g., '0.05,0.1'")
    p.add_argument("--sweep-bottleneck-dims", type=str, default="64",
                   help="Comma-separated bottleneck dims for sweep mode. e.g., '64,48'")
    p.add_argument("--sweep-eval-onnx", action="store_true",
                   help="Also evaluate ONNX for each sweep config (slower).")

    # ===== Mapping / Lookahead =====
    p.add_argument("--map-window", type=int, default=300)
    p.add_argument("--map-wz", type=float, default=3.0)

    p.add_argument("--look-m", type=int, default=10)
    p.add_argument("--delta-min", type=float, default=0.05)
    p.add_argument("--delta-max", type=float, default=10.0)
    p.add_argument("--look-cmd-w-now", type=float, default=0.80,
                   help="Weight for current command feed when estimating lookahead delta.")
    p.add_argument("--look-cmd-w-f1", type=float, default=0.15,
                   help="Weight for the first future command feed when estimating lookahead delta.")
    p.add_argument("--look-cmd-w-f2", type=float, default=0.05,
                   help="Weight for the second future command feed when estimating lookahead delta.")
    p.add_argument("--look-blend-min-weight", type=float, default=0.15,
                   help="Minimum weight kept on ramp-based delta when blending with geometric fallback.")
    p.add_argument("--look-track-err-ratio-ref", type=float, default=0.25,
                   help="Reference |feed_err_ratio| above which future/ramp lookahead confidence is reduced.")
    p.add_argument("--look-ramp-change-ref", type=float, default=300.0,
                   help="Reference |d_act_f| above which future/ramp lookahead confidence is reduced.")
    p.add_argument("--look-delta-ema-alpha", type=float, default=0.30,
                   help="EMA alpha for delta smoothing. Larger means more reactive.")
    p.add_argument("--look-delta-step-limit", type=float, default=0.20,
                   help="Maximum allowed per-step delta change after smoothing.")

    # ===== Exclude pairs from training (by filename tokens) =====
    p.add_argument("--exclude", type=str, default="O80,O81",
                   help="Comma-separated tokens. If a SIM/ACT filepath contains any token, that pair is excluded from training/val split. Set empty string to disable.")

    # ===== SIM→Load delay compensation (τ-ms interpolation) =====
    p.add_argument("--lag-auto", action="store_true", help="Estimate a global lag (in samples) where SIM (MRR/torque) leads actLoad, then apply it to align features.")
    p.add_argument("--no-lag-auto", dest="lag_auto", action="store_false", help="Disable auto lag estimation (default: enabled).")
    p.set_defaults(lag_auto=True)
    p.add_argument("--lag-max", type=int, default=25, help="Max lag (samples) to search when --lag-auto is enabled.")
    p.add_argument("--lag-method", type=str, default="combo_d1", choices=["torque","mrr","combo","torque_d1","mrr_d1","combo_d1"], help="Which SIM feature to use for lag estimation. 'combo' uses (MRR+Torque)/2 correlation.")
    p.add_argument("--lag-samples", type=int, default=-1, help="Override lag samples. -1 = auto (if enabled), 0 = disable.")
    p.add_argument("--tau-ms", type=float, default=180.0, help="Delay compensation τ in ms. Recommended default=180. Set <0 to auto (lag_samples*dt_median_ms).")
    p.add_argument("--dt-fallback-ms", type=float, default=60.0, help="Fallback dt (ms) if dt cannot be estimated reliably from act timestamps/intervals.")

    # ===== Residual feedback =====
    p.add_argument("--resid-W", type=int, default=50)
    p.add_argument("--resid-ki", type=float, default=0.01)
    p.add_argument("--resid-kp", type=float, default=0.02)
    p.add_argument("--resid-clip-i", type=float, default=0.1)
    p.add_argument("--resid-clip-p", type=float, default=0.1)

    # ===== scaler stability =====
    p.add_argument("--min-std", type=float, default=1e-8)
    # v12b: make raw output closer to persistence baseline
    p.add_argument("--dy-mean-zero", dest="dy_mean_zero", action="store_true",
                   help="Force dy_mean=0 so dy=0 corresponds to persistence (recommended).")
    p.add_argument("--no-dy-mean-zero", dest="dy_mean_zero", action="store_false",
                   help="Use empirical dy_mean from data (legacy).")
    p.set_defaults(dy_mean_zero=True)

    p.add_argument("--alpha-gate", dest="alpha_gate", action="store_true",
                   help="Learn alpha gate (0..1) to shrink dy when uncertain (recommended).")
    p.add_argument("--no-alpha-gate", dest="alpha_gate", action="store_false",
                   help="Disable alpha gate (legacy).")
    p.set_defaults(alpha_gate=True)

    p.add_argument("--alpha-l1", type=float, default=0.002, help="L1 penalty on alpha (persistence).")
    p.add_argument("--alpha-sup", type=float, default=0.0, help="L2 supervision on alpha to follow |dy| (helps raw become persist-like).")
    p.add_argument("--alpha-target-scale", type=float, default=1.0, help="In dy_norm units: |dy|>=scale -> alpha_target≈1 for alpha supervision.")
    p.add_argument("--alpha-warmup-epochs", type=int, default=0, help="Ramp alpha_l1 and dy_l2 from 0→full over N epochs.")
    p.add_argument("--dy-l2", type=float, default=0.002, help="L2 penalty on dy_norm magnitude.")
    p.add_argument("--dy-weight-k", type=float, default=4.0, help="Extra weight for large |dy_norm| events.")
    p.add_argument("--dy-weight-clip", type=float, default=4.5, help="Clip for |dy_norm| weighting.")
    p.add_argument("--dy-weight-pow", type=float, default=1.2, help="Exponent for dy event weighting. 1.0 linear, >1 emphasizes larger |dy|.")
    p.add_argument("--dy-clip-norm", type=float, default=8.0, help="Clamp model dy_norm output range.")
    p.add_argument("--dy-deadband-raw", type=float, default=0.010, help="Optional: set |dy_raw|<deadband to 0 in target (stabilize raw).")
    p.add_argument("--stable-lambda", type=float, default=0.015,
                   help="Penalty strength to push dy_hat toward 0 on stable segments (|dy_true_norm| small).")
    p.add_argument("--stable-dy-thresh-norm", type=float, default=0.22,
                   help="Stable threshold in normalized dy units: stable if |dy_true_norm| < thresh.")
    p.add_argument("--aircut-lambda", type=float, default=0.05,
                   help="Penalty strength to push dy_hat toward 0 when MRR is ~0 (air-cut).")
    p.add_argument("--aircut-mrr-eps", type=float, default=1e-9,
                   help="Air-cut threshold on raw MRR. If MRR <= eps, treated as air-cut.")
    p.add_argument("--persist-margin-lambda", type=float, default=0.45, help="Optional: penalize being worse than persist (dy_norm SSE margin loss).")
    p.add_argument('--persist-margin-warmup-epochs', type=int, default=12,help='Warmup epochs for persist-margin loss (0=disable warmup).')
    p.add_argument("--persist-margin", type=float, default=0.00, help="Margin for persist SSE loss in dy_norm space (>=0).")
    p.add_argument("--loss-type", type=str, default="huber", choices=["mse","huber"], help="Loss type for dy.")
    p.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta in dy_norm space.")


    return p.parse_args()


def _set_seed(seed: int):
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _run_one_experiment(
    tag: str,
    train_runs: List[RunData],
    val_runs: List[RunData],
    all_runs: List[RunData],
    scaler: FeatureScaler,
    resid_cfg: ResidualFeedbackConfig,
    base_args,
    head_type: str,
    head_dropout: float,
    bottleneck_dim: int,
    outdir: str,
    do_eval_onnx: bool
) -> Dict:
    train_cfg = TrainConfig(
        hidden_size=base_args.hidden_size,
        num_layers=base_args.num_layers,
        lr=base_args.lr,
        weight_decay=base_args.weight_decay,
        epochs=base_args.epochs,
        tbptt_steps=base_args.tbptt_steps,
        grad_clip=base_args.grad_clip,

        head_type=head_type,
        head_dropout=head_dropout,
        bottleneck_dim=bottleneck_dim,

        # v23: raw→persist helpers
        use_alpha_gate=base_args.alpha_gate,
        alpha_l1=base_args.alpha_l1,
        alpha_sup=base_args.alpha_sup,
        alpha_target_scale=base_args.alpha_target_scale,
        alpha_warmup_epochs=base_args.alpha_warmup_epochs,

        dy_l2=base_args.dy_l2,
        dy_weight_k=base_args.dy_weight_k,
        dy_weight_clip=base_args.dy_weight_clip,
        dy_weight_pow=base_args.dy_weight_pow,
        dy_clip_norm=base_args.dy_clip_norm,
        dy_deadband_raw=base_args.dy_deadband_raw,

        persist_margin_lambda=base_args.persist_margin_lambda,
        persist_margin_warmup_epochs=base_args.persist_margin_warmup_epochs,
        persist_margin=base_args.persist_margin,

        loss_type=base_args.loss_type,
        huber_delta=base_args.huber_delta,
        force_dy_mean_zero=base_args.dy_mean_zero,

        # early stopping / best selection
        early_stop=(not base_args.no_early_stop),
        es_patience=base_args.patience,
        es_min_delta=base_args.min_delta,
        es_eval_every=base_args.val_every,

        best_metric=base_args.best_metric,
        best_persist_penalty=base_args.best_persist_penalty,
        best_persist_margin=base_args.best_persist_margin,
        es_min_delta_score=base_args.min_delta_score,

        # finetune
        finetune=base_args.finetune,
        ft_epochs=base_args.ft_epochs,
        ft_lr_scale=base_args.ft_lr_scale,
        ft_weight_decay_scale=base_args.ft_weight_decay_scale,
        ft_dy_deadband_scale=base_args.ft_dy_deadband_scale,
        ft_persist_margin_lambda_scale=base_args.ft_persist_margin_lambda_scale,
        ft_dy_weight_k_scale=base_args.ft_dy_weight_k_scale,
        ft_dy_clip_norm_scale=base_args.ft_dy_clip_norm_scale,
        ft_stable_lambda_scale=base_args.ft_stable_lambda_scale,
        ft_aircut_lambda_scale=base_args.ft_aircut_lambda_scale,
        ft_patience=base_args.ft_patience,
        ft_eval_every=base_args.ft_val_every,
        best_focus=base_args.best_focus,
        best_event_dy_thresh_norm=base_args.best_event_dy_thresh_norm,
        best_event_weight=base_args.best_event_weight,

        stable_lambda=base_args.stable_lambda,
        stable_dy_thresh_norm=base_args.stable_dy_thresh_norm,
        aircut_lambda=base_args.aircut_lambda,
        aircut_mrr_eps=base_args.aircut_mrr_eps,
    )

    print(f"\n=== [SWEEP] {tag} ===")
    print(f"  head={head_type} dropout={head_dropout} bottleneck_dim={bottleneck_dim} hidden={train_cfg.hidden_size}")

    model = train_stateful_gru(
        runs=train_runs,
        val_runs=val_runs,
        scaler=scaler,
        resid_cfg=resid_cfg,
        cfg=train_cfg,
        entry_weight=1.3,
        entry_threshold=0.5
    )

    # Combine for overall evaluation (train+val) and infer input dims.
    all_runs_local = train_runs + val_runs
    A = all_runs_local[0].mapped.shape[1]
    B = all_runs_local[0].look.shape[1]
    input_size = A + B + 1

    onnx_name = os.path.join(outdir, f"cutload_stateful_gru_{_safe_tag(tag)}.onnx")
    export_onnx(model, onnx_name, input_size, train_cfg)

    torch_dump = os.path.join(outdir, f"debug_eval_torch_dump_{_safe_tag(tag)}.csv")
    onnx_dump  = os.path.join(outdir, f"debug_eval_onnx_dump_{_safe_tag(tag)}.csv")

    torch_res = eval_torch(model, all_runs_local, scaler, resid_cfg, train_cfg, dump_csv=torch_dump, dump_max=20000)
    print("[EVAL/TORCH]", torch_res)

    # validation-only metrics (for early-stop selection / generalization check)
    val_res = eval_torch(model, val_runs, scaler, resid_cfg, train_cfg, dump_csv=None, dump_max=0) if val_runs else None
    if val_res:
        print("[EVAL/VAL  ]", val_res)

    onnx_res = None
    if do_eval_onnx:
        onnx_res = eval_onnx(onnx_name, all_runs_local, scaler, resid_cfg, train_cfg,
                             input_size=torch_res["input_size"],
                             dump_csv=onnx_dump, dump_max=20000)
        print("[EVAL/ONNX ]", onnx_res)

    # summarize
    tr = torch_res.get("torch", {})
    summary = {
        "tag": tag,
        "head": head_type,
        "dropout": float(head_dropout),
        "bottleneck_dim": int(bottleneck_dim),
        "hidden": int(train_cfg.hidden_size),
        "val_RMSE": float(val_res["torch"]["RMSE"]) if val_res else float("nan"),
        "val_MAE": float(val_res["torch"]["MAE"]) if val_res else float("nan"),
        "val_R2": float(val_res["torch"]["R2"]) if val_res else float("nan"),
        "input_size": int(torch_res.get("input_size", input_size)),
        "RMSE": float(tr.get("RMSE", float("nan"))),
        "MAE": float(tr.get("MAE", float("nan"))),
        "R2": float(tr.get("R2", float("nan"))),
        "onnx_path": onnx_name,
    }
    return summary



def _safe_tag(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(s).strip())
    return s.strip("_") or "case"


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)


def _evaluate_named_cases(model,
                          onnx_path: str,
                          named_runs: List[Tuple[str, RunData]],
                          scaler: FeatureScaler,
                          resid_cfg: ResidualFeedbackConfig,
                          train_cfg: TrainConfig,
                          out_dir: str) -> Dict:
    cases: Dict[str, Dict] = {}
    for case_name, run in named_runs:
        tag = _safe_tag(case_name)
        torch_csv = os.path.join(out_dir, f"{tag}_torch_predictions.csv")
        onnx_csv = os.path.join(out_dir, f"{tag}_onnx_predictions.csv")

        torch_res = eval_torch(model, [run], scaler, resid_cfg, train_cfg,
                               dump_csv=torch_csv, dump_max=10**9)
        try:
            onnx_res = eval_onnx(onnx_path, [run], scaler, resid_cfg, train_cfg,
                                 input_size=int(torch_res.get("input_size", 0)),
                                 dump_csv=onnx_csv, dump_max=10**9)
        except Exception as e:
            onnx_res = {"error": str(e)}
            onnx_csv = None

        torch_case_metrics = torch_res.get("torch") or {}
        onnx_case_metrics = onnx_res.get("onnx") if isinstance(onnx_res, dict) else None

        cases[case_name] = {
            "summary": {
                "N": torch_case_metrics.get("N"),
                "MAE": torch_case_metrics.get("MAE"),
                "RMSE": torch_case_metrics.get("RMSE"),
                "MAPE (%)": torch_case_metrics.get("MAPE (%)"),
                "NRMSE (%)": torch_case_metrics.get("NRMSE (%)"),
                "R²": torch_case_metrics.get("R²", torch_case_metrics.get("R2")),
                "Peak error (%)": torch_case_metrics.get("Peak error (%)"),
                "Avg pred. time (ms)": torch_case_metrics.get("Avg pred. time (ms)"),
                "P95 pred. time (ms)": torch_case_metrics.get("P95 pred. time (ms)"),
            },
            "torch": torch_case_metrics,
            "onnx": onnx_case_metrics,
            "mean_baseline": torch_res.get("mean_baseline"),
            "persist_baseline": torch_res.get("persist_baseline"),
            "torch_event": torch_res.get("torch_event"),
            "persist_event": torch_res.get("persist_event"),
            "onnx_event": onnx_res.get("onnx_event") if isinstance(onnx_res, dict) else None,
            "artifacts": {
                "torch_predictions_csv": torch_csv,
                "onnx_predictions_csv": onnx_csv,
            },
        }
    return {"cases": cases}

def main():
    args = parse_args()
    _set_seed(args.seed)

    script_dir = _SCRIPT_DIR()
    sim_glob, act_glob, data_tag = _resolve_default_globs()
    paths_cfg = PathsConfig(sim_glob=sim_glob, act_glob=act_glob)
    print(f"[DATA] {data_tag}  sim_glob={paths_cfg.sim_glob}  act_glob={paths_cfg.act_glob}")

    run_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = _ensure_dir(os.path.join(script_dir, f"{run_name}_results"))

    out_onnx_path = os.path.join(output_dir, "cutload_stateful_gru.onnx")
    out_scaler_path = os.path.join(output_dir, "scaler_params.json")
    out_torch_dump = os.path.join(output_dir, "debug_eval_torch_dump.csv")
    out_onnx_dump  = os.path.join(output_dir, "debug_eval_onnx_dump.csv")

    map_cfg = MappingConfig(window=args.map_window, wz=args.map_wz)
    look_cfg = LookAheadConfig(
        m=args.look_m,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        cmd_w_now=args.look_cmd_w_now,
        cmd_w_f1=args.look_cmd_w_f1,
        cmd_w_f2=args.look_cmd_w_f2,
        blend_min_weight=args.look_blend_min_weight,
        track_err_ratio_ref=args.look_track_err_ratio_ref,
        ramp_change_ref=args.look_ramp_change_ref,
        delta_ema_alpha=args.look_delta_ema_alpha,
        delta_step_limit=args.look_delta_step_limit,
    )
    resid_cfg = ResidualFeedbackConfig(W=args.resid_W, ki=args.resid_ki, kp=args.resid_kp,
                                       clip_i=args.resid_clip_i, clip_p=args.resid_clip_p, use_mean=True)

    pairs = find_pairs_by_feed_depth(paths_cfg)

    # ===== Exclude specific runs (e.g., O80/O81) from TRAIN/VAL build =====
    excl = str(getattr(args, 'exclude', '') or '').strip()
    excl_tokens = [t.strip() for t in excl.split(',') if t.strip()]
    if excl_tokens:
        _before = len(pairs)
        def _has_tok(path: str) -> bool:
            pth = str(path)
            return any(tok in pth for tok in excl_tokens)
        pairs = [p for p in pairs if (not _has_tok(p[1]) and not _has_tok(p[2]))]
        _removed = _before - len(pairs)
        if _removed > 0:
            print(f"[EXCLUDE] removed {_removed}/{_before} pairs by tokens={excl_tokens}")

    if not pairs:
        raise RuntimeError(
            "No (sim, act) pairs found.\n"
            f"  sim_glob: {paths_cfg.sim_glob}\n"
            f"  act_glob: {paths_cfg.act_glob}\n"
            "스크립트와 같은 폴더에 아래 구조가 있는지 확인하세요.\n"
            "  ./Sim_20mm/*.csv\n"
            "  ./Act_20mm/*.csv\n"
            "(또는 12mm의 경우 Sim_12mm/Act_12mm)\n"
            "※ 실행 위치가 달라도 자동 탐색하지만, 폴더명이 다르면 찾지 못합니다."
        )

    print("[PAIRS]")
    for pref, sp, ap in pairs:
        print(f"  {pref}: sim={sp} act={ap}")


    # 1) split pairs into train/val (by prefixes)
    import random as _random
    pairs_shuffled = list(pairs)
    _random.Random(args.seed).shuffle(pairs_shuffled)

    val_ratio = float(args.val_ratio)
    if val_ratio < 0.0:
        val_ratio = 0.0
    if val_ratio > 0.9:
        val_ratio = 0.9

    n_val = 0
    if val_ratio > 0.0 and len(pairs_shuffled) >= 3:
        n_val = int(round(len(pairs_shuffled) * val_ratio))
        n_val = max(1, min(n_val, len(pairs_shuffled) - 1))

    val_pairs = pairs_shuffled[:n_val]
    train_pairs = pairs_shuffled[n_val:]

    print(f"[SPLIT] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} (val_ratio={val_ratio})")
    if val_pairs:
        print("  [VAL ] " + ", ".join([p[0] for p in val_pairs]))
    print("  [TRAIN] " + ", ".join([p[0] for p in train_pairs]))

    # 2) build raw runs
    train_runs: List[RunData] = []
    val_runs: List[RunData] = []

    def _load_one(pref: str, sim_csv: str, act_csv: str) -> RunData:
        print(f"\n[LOAD] prefix={pref}")
        sim = load_sim(sim_csv)
        print(f"  sim train s-range: [{sim.train_start_s:.3f}, {sim.train_end_s:.3f}] "
              f"(N={sim.xyz.shape[0]}, blocks={len(sim.block_indices)})")

        run = build_run_from_files(sim, act_csv, map_cfg, look_cfg)
        cut0 = _first_cut_index(run.train_mask)
        pre_cut = int(cut0) if cut0 is not None else -1
        print(f"  run N={run.y_raw.shape[0]}  train_mask={int(run.train_mask.sum())}/{len(run.train_mask)}  cut0={pre_cut}")
        return run

    for pref, sim_csv, act_csv in train_pairs:
        train_runs.append(_load_one(pref, sim_csv, act_csv))
    for pref, sim_csv, act_csv in val_pairs:
        val_runs.append(_load_one(pref, sim_csv, act_csv))


    target_runs: List[RunData] = []
    target_run_names: List[str] = []
    target_run_specs: List[Tuple[str, str, str]] = []

    # hard-coded O80/O81 test sets
    for case_name, sim_csv, act_csv in _resolve_hardcoded_test_cases(script_dir):
        target_run_specs.append((case_name, sim_csv, act_csv))

    # optional extra target runs from CLI
    t_sims = args.target_sim if isinstance(args.target_sim, list) else [str(args.target_sim)]
    t_acts = args.target_act if isinstance(args.target_act, list) else [str(args.target_act)]
    if len(t_sims) == 1 and ("," in t_sims[0]):
        t_sims = [s.strip() for s in t_sims[0].split(",") if s.strip()]
    if len(t_acts) == 1 and ("," in t_acts[0]):
        t_acts = [s.strip() for s in t_acts[0].split(",") if s.strip()]
    K = min(len(t_sims), len(t_acts))
    for k in range(K):
        if t_sims[k] and t_acts[k]:
            target_run_specs.append((f"TARGET_EXTRA{k+1}", t_sims[k], t_acts[k]))

    seen_target_keys = set()
    for pref, sim_csv, act_csv in target_run_specs:
        key = (os.path.normcase(str(sim_csv)), os.path.normcase(str(act_csv)))
        if key in seen_target_keys:
            continue
        seen_target_keys.add(key)
        target_runs.append(_load_one(pref, sim_csv, act_csv))
        target_run_names.append(pref)

    # 2.5) (NEW) Estimate & apply SIM→Load delay τ (ms) with interpolation
    # Observed: SIM MRR/Torque rises first, measured Cutting Load follows after a short delay.
    # We estimate lag (samples) on TRAIN runs, convert to τ-ms using median dt, and apply τ-ms interpolation to SIM features.
    lag_samples_used = 0
    tau_ms_used = 0.0
    dt_median_ms_used = 60.0
    lag_method = str(getattr(args, 'lag_method', 'combo'))
    lag_max = int(getattr(args, 'lag_max', 0))
    tau_override = float(getattr(args, 'tau_ms', -1.0))

    # (A) Choose lag in samples (manual or auto)
    if int(getattr(args, 'lag_samples', -1)) >= 0:
        lag_samples_used = int(args.lag_samples)
        print(f"[TAU] override lag_samples={lag_samples_used}")
    elif bool(getattr(args, 'lag_auto', True)) and lag_max > 0:
        per = []
        print(f"[TAU] auto-estimating lag (0..{lag_max}) using method='{lag_method}' on TRAIN runs...")
        for i, r in enumerate(train_runs):
            L, st = estimate_lag_samples_from_run(r, max_lag=lag_max, method=lag_method, mrr_eps=float(args.aircut_mrr_eps))
            per.append(L)
            print(f"  run#{i+1:02d}: lag={L}  corr={st.get('corr', float('nan')):.4f}  n_cut={int(st.get('n',0))}")
        if per:
            import numpy as _np
            lag_samples_used = int(_np.median(_np.array(per, dtype=_np.int32)))
        print(f"[TAU] selected global lag_samples={lag_samples_used} (median of train runs)")

    # (B) Estimate dt_median_ms on TRAIN runs (for τ conversion)
    try:
        import numpy as _np
        dts = [_dt_median_ms(r, fallback=float(getattr(args,'dt_fallback_ms',60.0))) for r in train_runs]
        dt_median_ms_used = float(_np.median(_np.array(dts, dtype=_np.float64))) if dts else float(getattr(args,'dt_fallback_ms',60.0))
    except:
        dt_median_ms_used = float(getattr(args,'dt_fallback_ms',60.0))

    # (C) Choose τ-ms
    if tau_override >= 0.0:
        tau_ms_used = float(tau_override)
        if lag_samples_used <= 0:
            # for reporting only
            lag_samples_used = int(round(tau_ms_used / max(1e-9, dt_median_ms_used)))
        print(f"[TAU] override tau_ms={tau_ms_used:.3f} ms (dt_median={dt_median_ms_used:.3f} ms, lag≈{lag_samples_used} samples)")
    elif lag_samples_used > 0:
        tau_ms_used = float(lag_samples_used) * float(dt_median_ms_used)
        print(f"[TAU] tau_ms={tau_ms_used:.3f} ms from lag_samples={lag_samples_used} * dt_median={dt_median_ms_used:.3f} ms")

    # (D) Apply τ-ms interpolation shift to SIM features
    if tau_ms_used > 0.0:
        print(f"[TAU] applying tau_ms={tau_ms_used:.3f} ms to train/val/target runs (interpolate SIM features at t-τ)")
        for r in (train_runs + val_runs + target_runs):
            apply_sim_tau_to_run(r, tau_ms_used, mrr_eps=float(args.aircut_mrr_eps))

    # all_runs will be set after scaler is applied
    # 2) fit scaler and apply (once)
    fit_runs = train_runs
    if args.scaler_fit_target and target_runs:
        fit_runs = train_runs + target_runs

    scaler = FeatureScaler.fit(fit_runs, eps=args.min_std, force_dy_mean_zero=args.dy_mean_zero)
    for r in (train_runs + val_runs + target_runs):
        scaler.apply_run(r)

    named_target_runs = list(zip(target_run_names, target_runs))
    all_runs = train_runs + val_runs + target_runs

    with open(out_scaler_path, "w", encoding="utf-8") as f:
        _sc = scaler.to_json()
        _sc["lag_samples"] = int(lag_samples_used)
        _sc["lag_method"] = str(lag_method)
        _sc["lag_max"] = int(lag_max)
        _sc["dt_median_ms"] = float(dt_median_ms_used)
        _sc["tau_ms"] = float(tau_ms_used)
        _sc["dy_clip_norm"] = float(args.dy_clip_norm)
        json.dump(_sc, f, ensure_ascii=False, indent=2)
    print(f"[SCALER] saved: {out_scaler_path}")

    # 3) train & eval
    if not args.sweep:
        train_cfg = TrainConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            tbptt_steps=args.tbptt_steps,
            grad_clip=args.grad_clip,

            head_type=args.head,
            head_dropout=args.head_dropout,
            bottleneck_dim=args.bottleneck_dim,

            use_alpha_gate=args.alpha_gate,
            alpha_l1=args.alpha_l1,
            alpha_sup=args.alpha_sup,
            alpha_target_scale=args.alpha_target_scale,
            alpha_warmup_epochs=args.alpha_warmup_epochs,

            dy_l2=args.dy_l2,
            dy_weight_k=args.dy_weight_k,
            dy_weight_clip=args.dy_weight_clip,
            dy_weight_pow=args.dy_weight_pow,
            dy_clip_norm=args.dy_clip_norm,
            dy_deadband_raw=args.dy_deadband_raw,

            persist_margin_lambda=args.persist_margin_lambda,
            persist_margin_warmup_epochs=args.persist_margin_warmup_epochs,
            persist_margin=args.persist_margin,

            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            force_dy_mean_zero=args.dy_mean_zero,

            early_stop=(not args.no_early_stop),
            es_patience=args.patience,
            es_min_delta=args.min_delta,
            es_eval_every=args.val_every,

            best_metric=args.best_metric,
            best_persist_penalty=args.best_persist_penalty,
            best_persist_margin=args.best_persist_margin,
            es_min_delta_score=args.min_delta_score,

            finetune=args.finetune,
            ft_epochs=args.ft_epochs,
            ft_lr_scale=args.ft_lr_scale,
            ft_weight_decay_scale=args.ft_weight_decay_scale,
            ft_dy_deadband_scale=args.ft_dy_deadband_scale,
            ft_persist_margin_lambda_scale=args.ft_persist_margin_lambda_scale,
            ft_dy_weight_k_scale=args.ft_dy_weight_k_scale,
            ft_dy_clip_norm_scale=args.ft_dy_clip_norm_scale,
            ft_stable_lambda_scale=args.ft_stable_lambda_scale,
            ft_aircut_lambda_scale=args.ft_aircut_lambda_scale,
            ft_patience=args.ft_patience,
            ft_eval_every=args.ft_val_every,

            best_focus=args.best_focus,
            best_event_dy_thresh_norm=args.best_event_dy_thresh_norm,
            best_event_weight=args.best_event_weight,

            stable_lambda=args.stable_lambda,
            stable_dy_thresh_norm=args.stable_dy_thresh_norm,
            aircut_lambda=args.aircut_lambda,
            aircut_mrr_eps=args.aircut_mrr_eps,

            # v26
            reset_on_aircut=(not args.no_reset_on_aircut),
            reset_on_aircut_hold_persist=(not args.no_aircut_hold_persist),
            aircut_reset_min_steps=int(getattr(args, 'aircut_reset_min_steps', 1)),
            target_score_weight=float(args.target_score_weight),
            val_calib=(not args.no_val_calib),
            target_calib=(not args.no_target_calib),
            target_calib_only=(not args.target_calib_full),
            target_calib_epochs=int(args.target_calib_epochs),
            target_calib_lr_scale=float(args.target_calib_lr_scale),
            target_calib_gain_min=float(args.target_calib_gain_min),
            target_calib_gain_max=float(args.target_calib_gain_max),
            target_calib_bias_abs=float(args.target_calib_bias_abs),
            target_finetune=(not args.no_target_finetune),
            target_ft_epochs=int(args.target_ft_epochs),
            target_ft_lr_scale=float(args.target_ft_lr_scale),
            target_freeze_gru=(not args.no_target_freeze_gru),
        )

        model = train_stateful_gru(
            runs=train_runs,
            val_runs=val_runs,
            target_runs=target_runs,
            scaler=scaler,
            resid_cfg=resid_cfg,
            cfg=train_cfg,
            entry_weight=1.3,
            entry_threshold=0.5
        )


        # v3: write dy_gain/dy_bias (calibrated) to scaler_params.json for C# runtime parity
        try:
            with open(out_scaler_path, "r", encoding="utf-8") as _f:
                _scj = json.load(_f)
            if hasattr(model, "dy_gain"):
                _scj["dy_gain"] = float(model.dy_gain.detach().cpu().numpy().reshape(-1)[0])
            if hasattr(model, "dy_bias"):
                _scj["dy_bias"] = float(model.dy_bias.detach().cpu().numpy().reshape(-1)[0])
            _scj["dy_clip_norm"] = float(args.dy_clip_norm)
            with open(out_scaler_path, "w", encoding="utf-8") as _f:
                json.dump(_scj, _f, ensure_ascii=False, indent=2)
            print(f"[SCALER] updated: {out_scaler_path} (dy_gain/dy_bias/dy_clip_norm)")
        except Exception as e:
            print("[SCALER] warning: failed to update dy_gain/dy_bias:", e)

        A = all_runs[0].mapped.shape[1]
        B = all_runs[0].look.shape[1]
        input_size = A + B + 1
        export_onnx(model, out_onnx_path, input_size, train_cfg)

        torch_dump = out_torch_dump
        onnx_dump  = out_onnx_dump

        torch_res = eval_torch(model, all_runs, scaler, resid_cfg, train_cfg, dump_csv=torch_dump, dump_max=20000)
        print("[EVAL/TORCH]", torch_res)

        # validation-only metrics (if validation split was used)
        if val_runs:
            val_res = eval_torch(model, val_runs, scaler, resid_cfg, train_cfg, dump_csv=None, dump_max=0)
            print("[EVAL/VAL  ]", val_res)

        try:
            onnx_res = eval_onnx(out_onnx_path, all_runs, scaler, resid_cfg, train_cfg,
                                 input_size=torch_res["input_size"],
                                 dump_csv=onnx_dump, dump_max=20000)
            print("[EVAL/ONNX ]", onnx_res)
        except ImportError as e:
            onnx_res = None
            print("[EVAL/ONNX ] skipped (onnxruntime not installed):", e)



        train_summary_path = os.path.join(output_dir, "training_summary.json")
        test_summary_path = os.path.join(output_dir, "o80_o81_test_summary.json")

        val_res_local = eval_torch(model, val_runs, scaler, resid_cfg, train_cfg, dump_csv=None, dump_max=0) if val_runs else None
        case_summary = _evaluate_named_cases(model, out_onnx_path, named_target_runs, scaler, resid_cfg, train_cfg, output_dir) if named_target_runs else {"cases": {}}

        train_summary = {
            "script": os.path.basename(__file__),
            "paths": {
                "sim_glob": paths_cfg.sim_glob,
                "act_glob": paths_cfg.act_glob,
                "hardcoded_targets": [
                    {"name": n, "sim": s, "act": a} for (n, s, a) in target_run_specs
                ],
            },
            "split": {
                "excluded_tokens": excl_tokens,
                "train_prefixes": [p[0] for p in train_pairs],
                "val_prefixes": [p[0] for p in val_pairs],
                "target_prefixes": list(target_run_names),
            },
            "tau": {
                "lag_samples": int(lag_samples_used),
                "lag_method": str(lag_method),
                "lag_max": int(lag_max),
                "dt_median_ms": float(dt_median_ms_used),
                "tau_override_ms": float(tau_override),
                "tau_ms": float(tau_ms_used),
            },
            "train_cfg": dict(vars(train_cfg)),
            "overall_eval": {
                "torch": torch_res,
                "onnx": onnx_res,
                "val": val_res_local,
            },
            "artifacts": {
                "output_dir": output_dir,
                "onnx": out_onnx_path,
                "scaler": out_scaler_path,
                "torch_dump": torch_dump,
                "onnx_dump": onnx_dump,
                "training_summary_json": train_summary_path,
                "test_summary_json": test_summary_path,
            },
        }
        _write_json(train_summary_path, train_summary)
        _write_json(test_summary_path, case_summary)

        if named_target_runs:
            for case_name, case_payload in case_summary.get("cases", {}).items():
                print(f"[TEST/{case_name}/TORCH]", case_payload.get("torch"))
                if case_payload.get("onnx") is not None:
                    print(f"[TEST/{case_name}/ONNX ]", case_payload.get("onnx"))
            print(f"[JSON] saved: {test_summary_path}")
        print(f"[JSON] saved: {train_summary_path}")

        if isinstance(onnx_res, dict) and "onnx" in onnx_res and onnx_res["onnx"]["N"] > 0:
            compare_torch_onnx_firstK(torch_dump, onnx_dump, k=2000)
        return

    # ===== SWEEP MODE =====
    sweep_dropouts = _parse_csv_floats(args.sweep_dropouts)
    if not sweep_dropouts:
        sweep_dropouts = [0.05, 0.1]
    sweep_bdims = _parse_csv_ints(args.sweep_bottleneck_dims)
    if not sweep_bdims:
        sweep_bdims = [64]
    base_out_arg = args.sweep_outdir
    if not os.path.isabs(base_out_arg):
        base_out_arg = os.path.join(output_dir, base_out_arg)
    base_out = _ensure_dir(base_out_arg)
    outdir = _ensure_dir(os.path.join(base_out, _timestamp()))
    print(f"\n[SWEEP] outdir={outdir}")
    # save scaler copy inside outdir
    with open(os.path.join(outdir, "scaler_params.json"), "w", encoding="utf-8") as f:
        json.dump(scaler.to_json(), f, ensure_ascii=False, indent=2)

    configs = []
    # baseline as reference (dropout fixed 0.2)
    configs.append(("baseline_do0.2", "baseline", 0.2, args.bottleneck_dim))
    # residual over dropouts
    for d in sweep_dropouts:
        configs.append((f"residual_do{d}", "residual", d, args.bottleneck_dim))
    # bottleneck(64) over dropouts
    for bd in sweep_bdims:
        for d in sweep_dropouts:
            configs.append((f"bottleneck{bd}_do{d}", "bottleneck", d, bd))

    results: List[Dict] = []
    best = None  # (rmse, result)
    for tag, head_type, head_do, bd in configs:
        r = _run_one_experiment(
            tag=tag,
            train_runs=train_runs,
            val_runs=val_runs,
            all_runs=all_runs,
            scaler=scaler,
            resid_cfg=resid_cfg,
            base_args=args,
            head_type=head_type,
            head_dropout=head_do,
            bottleneck_dim=bd,
            outdir=outdir,
            do_eval_onnx=args.sweep_eval_onnx
        )
        results.append(r)
        metric_key = "val_RMSE" if (val_runs and (r.get("val_RMSE") == r.get("val_RMSE"))) else "RMSE"
        metric = r.get(metric_key, float("nan"))
        if best is None or (metric == metric and metric < best[0]):  # metric==metric filters NaN
            best = (metric, r)

    # write summary csv
    csv_path = os.path.join(outdir, "sweep_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "head", "dropout", "bottleneck_dim", "hidden", "input_size", "val_RMSE", "val_MAE", "val_R2", "RMSE", "MAE", "R2", "onnx_path"])
        for r in results:
            w.writerow([r["tag"], r["head"], r["dropout"], r["bottleneck_dim"], r["hidden"], r["input_size"],
                        r.get("val_RMSE", float("nan")), r.get("val_MAE", float("nan")), r.get("val_R2", float("nan")),
                        r["RMSE"], r["MAE"], r["R2"], r["onnx_path"]])
    print(f"\n[SWEEP] saved: {csv_path}")

    # print top-3 by (val_RMSE if available else RMSE)
    key_name = "val_RMSE" if (val_runs and any((r.get("val_RMSE") == r.get("val_RMSE")) for r in results)) else "RMSE"
    results_sorted = sorted([r for r in results if r.get(key_name) == r.get(key_name)], key=lambda x: x.get(key_name))
    print(f"\n[SWEEP] Top by {key_name}:")
    for i, r in enumerate(results_sorted[:3]):
        if key_name == "val_RMSE":
            print(f"  {i+1}) {r['tag']}: val_RMSE={r.get('val_RMSE', float('nan')):.6f}, RMSE={r['RMSE']:.6f}, MAE={r['MAE']:.6f}, R2={r['R2']:.6f}")
        else:
            print(f"  {i+1}) {r['tag']}: RMSE={r['RMSE']:.6f}, MAE={r['MAE']:.6f}, R2={r['R2']:.6f}")

    # copy best to default name for immediate use
    if best is not None:
        best_r = best[1]
        shutil.copyfile(best_r["onnx_path"], out_onnx_path)
        shutil.copyfile(os.path.join(outdir, "scaler_params.json"), out_scaler_path)
        print(f"\n[SWEEP] BEST => {best_r['tag']} (RMSE={best_r['RMSE']:.6f})")
        print(f"        copied to: {out_onnx_path} / {out_scaler_path}")

    print("\n[SWEEP] Done.")
    print("  Tip) 스윕에서 bottleneck(64) + dropout(0.05~0.1)이 v8보다 좋아지는지 우선 확인하고,")
    print("       그래도 부족하면 hidden_size=128 + head=128->64->1 쪽을 다음 단계로 가는 게 효율적입니다.")


if __name__ == "__main__":
    main()
