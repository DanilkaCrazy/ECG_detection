"""Stage 3.6.10 — registry of the five physiological feature groups.

Existing feature arrays are reused instead of recomputed; new groups A/E are
computed from raw signals by ``stage3610_extract_features.py``.  Every feature
carries its group so ablation E0-E8 can be assembled deterministically.
"""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["GROUPS", "select_group_columns", "load_unified_features"]

GROUPS = {
    "A": "rhythm",
    "B": "atrial_activity",
    "C": "time_frequency",
    "D": "morphology",
    "E": "interlead",
}

# Patterns selecting the physiologically meaningful subset of the 431-column
# residual feature array (not the whole diagnostic block).
B_PATTERNS = (
    "dominant_freq", "spectral_entropy", "spectral_concentration",
    "peak_prominence", "peak_width", "periodicity", "cycle_morph_consistency",
    "atrial_cycle_cv", "rr_cv", "spectral_flatness",
)


def select_group_columns(names: list[str]) -> dict[str, list[int]]:
    """Map group letters to column indices in the unified feature matrix."""
    lookup = {name: i for i, name in enumerate(names)}
    out: dict[str, list[int]] = {g: [] for g in GROUPS}
    for name, i in lookup.items():
        if name.startswith("rr_") or name.startswith("rhythm_"):
            out["A"].append(i)
        elif name.startswith("stft_") or name.startswith("tf_"):
            out["C"].append(i)
        elif name.startswith("interlead_"):
            out["E"].append(i)
        elif name.startswith("org_") or name.startswith("morph_"):
            out["D"].append(i)
        elif any(p in name for p in B_PATTERNS):
            out["B"].append(i)
    for g in out:
        out[g] = sorted(set(out[g]))
    return out


def load_unified_features(path: pathlib.Path) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Load the Stage 3.6.10 unified matrix and its group index map."""
    X = np.load(path / "X.npy", mmap_mode="r")
    names = pd.read_csv(path / "feature_names.csv")["name"].astype(str).tolist()
    if X.shape[1] != len(names):
        raise ValueError("feature matrix / names mismatch")
    return X, names, select_group_columns(names)
