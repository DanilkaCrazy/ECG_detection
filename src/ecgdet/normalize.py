"""Robust normalization utilities for ECG.

Following "Mathews/robust stats" guidance: prefer median/MAD, IQR-based scaling,
and quantile clipping over naive min-max. Critically, all statistics are
computed on the *training* partition and broadcast to validation/test to
avoid data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "RobustNormConfig",
    "fit_robust_norm",
    "apply_robust_norm",
    "winsorize",
    "per_lead_stats",
]


@dataclass
class RobustNormConfig:
    method: Literal[
        "none",
        "zscore",
        "robust_z_mad",
        "robust_iqr",
        "median_mad",
        "quantile_robust",
        "per_record_robust_z",
    ] = "robust_z_mad"
    eps: float = 1e-8
    winsor_low: float = 0.001
    winsor_high: float = 0.999
    use_winsor: bool = True
    # For per-record normalization there is nothing to "fit" globally.
    global_fit: bool = True


def winsorize(x: np.ndarray, low: float = 0.001, high: float = 0.999) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return x
    lo = np.quantile(x, low)
    hi = np.quantile(x, high)
    return np.clip(x, lo, hi)


def per_lead_stats(x: np.ndarray, eps: float = 1e-8) -> dict:
    """x shape (n_samples, n_leads)."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    q1 = np.quantile(x, 0.25, axis=0)
    q3 = np.quantile(x, 0.75, axis=0)
    iqr = q3 - q1
    return {
        "mean": mean,
        "std": np.where(std < eps, 1.0, std),
        "median": median,
        "mad": np.where(mad < eps, 1.0, mad),
        "q1": q1,
        "q3": q3,
        "iqr": np.where(iqr < eps, 1.0, iqr),
    }


def fit_robust_norm(
    X: np.ndarray,
    config: RobustNormConfig | None = None,
) -> dict:
    """Fit normalization statistics on training data only.

    X can be (n_records_or_samples, n_leads) or higher; we flatten to 2D per-lead.
    """
    config = config or RobustNormConfig()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        # (n_records, n_samples, n_leads)
        X2 = X.reshape(-1, X.shape[2])
    elif X.ndim == 2:
        X2 = X
    else:
        raise ValueError(f"unsupported shape {X.shape}")
    stats = per_lead_stats(X2, eps=config.eps)
    return {"config": config, "stats": stats}


def apply_robust_norm(
    X: np.ndarray,
    fitted: dict,
    fit_on_split: bool = False,
) -> np.ndarray:
    """Apply fitted (train-only) normalization."""
    cfg: RobustNormConfig = fitted["config"]
    st = fitted["stats"]
    X = np.asarray(X, dtype=np.float64).copy()
    eps = cfg.eps

    def _norm2d(x2: np.ndarray) -> np.ndarray:
        if cfg.use_winsor and cfg.method not in ("none",):
            x2 = np.column_stack(
                [winsorize(x2[:, i], cfg.winsor_low, cfg.winsor_high) for i in range(x2.shape[1])]
            )
        if cfg.method == "none":
            return x2
        if cfg.method == "zscore":
            return (x2 - st["mean"]) / st["std"]
        if cfg.method in ("robust_z_mad", "median_mad"):
            # (x - median) / (1.4826 * MAD) approx normal-consistent; use 1/MAD
            return (x2 - st["median"]) / (1.482602218505602 * st["mad"] + eps)
        if cfg.method == "robust_iqr":
            return (x2 - st["median"]) / st["iqr"]
        if cfg.method == "quantile_robust":
            # median/IQR with winsorization already applied
            return (x2 - st["median"]) / st["iqr"]
        if cfg.method == "per_record_robust_z":
            # record-wise robust scaling (no global fit required)
            med = np.median(x2, axis=0)
            mad = np.median(np.abs(x2 - med), axis=0)
            mad = np.where(mad < eps, 1.0, mad)
            if cfg.use_winsor:
                x2 = np.column_stack(
                    [winsorize(x2[:, i], cfg.winsor_low, cfg.winsor_high) for i in range(x2.shape[1])]
                )
            return (x2 - med) / (1.482602218505602 * mad + eps)
        raise ValueError(f"unknown method {cfg.method}")

    if X.ndim == 2:
        return _norm2d(X)
    if X.ndim == 3:
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            out[i] = _norm2d(X[i])
        return out
    raise ValueError(f"unsupported shape {X.shape}")
