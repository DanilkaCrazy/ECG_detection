"""Stage 3.6.10 — Group B: atrial activity features (selection over existing array).

The 431-column ``features_atrial_residual.npy`` already contains the physical
descriptors.  This module selects the interpretable subset and attaches a
cancellation-quality column so an unreliable QRST suppression can be masked
rather than silently read as atrial activity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ATRIAL_PATTERNS", "select_atrial_features"]

ATRIAL_PATTERNS = (
    "dominant_freq", "dominant_freq_cv", "dominant_freq_stability",
    "spectral_entropy", "spectral_concentration", "spectral_flatness",
    "peak_prominence", "peak_width", "peak_area_ratio",
    "periodicity", "cycle_morph_consistency", "atrial_cycle_cv",
    "rr_cv", "atrial_to_ventricular_rate",
)


def select_atrial_features(names: list[str]) -> list[int]:
    """Return indices of atrial-activity descriptors (excluding diagnostics)."""
    out = []
    for i, name in enumerate(names):
        low = name.lower()
        if low.startswith("diagnostic") or "qrs_supp" in low:
            continue
        if any(p in low for p in ATRIAL_PATTERNS):
            out.append(i)
    return out
