"""Stage 3.6.15 — explicit, auditable ECG normalisation variants.

For an array shaped (N, leads, samples) the OLD code normalised over ``axis=1``
(leads), subtracting the across-lead mean at each time point.  That is a
spatial transform, not per-lead temporal normalisation.  Variants below make
the choice explicit and keep train-fitted statistics separate.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_per_lead_time", "normalize_interlead", "normalize_robust",
    "fit_train_stats", "apply_train_stats", "NORMALIZATION_VARIANTS",
]

NORMALIZATION_VARIANTS = ("per_lead_time", "interlead", "robust", "none")


def normalize_per_lead_time(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """z-score EACH lead over TIME within each record (axis=-1)."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)


def normalize_interlead(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Legacy research variant: subtract the across-lead mean at each sample."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def normalize_robust(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Median / IQR per lead over time within each record."""
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x, axis=-1, keepdims=True)
    q1, q3 = np.percentile(x, [25, 75], axis=-1)
    iqr = np.maximum(q3 - q1, eps)[:, :, None]
    return np.clip((x - med) / iqr, -20, 20).astype(np.float32)


def fit_train_stats(x_train: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-lead robust statistics from TRAIN rows only."""
    x = np.asarray(x_train, dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    q1, q3 = np.nanpercentile(x, [25, 75], axis=0)
    scale = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return {"median": med.astype(np.float32), "scale": scale.astype(np.float32)}


def apply_train_stats(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Apply frozen train statistics to ANY split — no per-split refitting."""
    x = np.asarray(x, dtype=np.float32)
    out = (x - stats["median"]) / stats["scale"]
    return np.clip(np.nan_to_num(out), -20, 20).astype(np.float32)
