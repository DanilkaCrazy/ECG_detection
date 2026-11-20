"""Stage 3.6.15 — physical signal characteristics BEFORE harmonisation.

Only descriptive statistics; no claim that high-frequency energy is noise.
"""
from __future__ import annotations

import numpy as np

__all__ = ["audit_signal", "summarize_source_signals"]


def audit_signal(x: np.ndarray, fs: float, lead_names: list[str]) -> dict:
    """Summarise amplitude, quality and spectral properties of a raw ECG."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signal must be (leads, samples)")
    fs = float(fs)
    finite = np.isfinite(x)
    clean = np.where(finite, x, np.nan)
    nan_frac = float(np.mean(np.isnan(x)))
    inf_frac = float(np.mean(np.isinf(x)))
    filled = np.nan_to_num(x)

    mean = float(np.mean(filled)) if filled.size else 0.0
    std = float(np.std(filled)) if filled.size else 0.0
    rms = float(np.sqrt(np.mean(filled ** 2))) if filled.size else 0.0
    zero_frac = float(np.mean(np.abs(filled) < 1e-8))
    absx = np.abs(filled)
    p95 = float(np.percentile(absx, 95)) if absx.size else 0.0
    p99 = float(np.percentile(absx, 99)) if absx.size else 0.0
    clip = float(np.mean(absx >= 0.95 * max(p99, 1e-12)))

    # Baseline drift: std of a 1-second moving average, median over leads.
    win = max(2, int(fs))
    drift = []
    hf = []
    for row in filled:
        if len(row) >= win:
            ma = np.convolve(row, np.ones(win) / win, mode="valid")
            drift.append(float(np.std(ma)))
        spec = np.abs(np.fft.rfft(row - row.mean()))
        fr = np.fft.rfftfreq(len(row), 1 / fs)
        total = float(np.sum(spec ** 2))
        hf.append(float(np.sum(spec[fr > 40] ** 2) / total) if total > 0 else 0.0)

    return {
        "n_leads": int(x.shape[0]), "n_samples": int(x.shape[1]), "fs": fs,
        "mean": mean, "std": std, "rms": rms,
        "min": float(np.min(filled)) if filled.size else 0.0,
        "max": float(np.max(filled)) if filled.size else 0.0,
        "p95_abs": p95, "p99_abs": p99,
        "nan_fraction": nan_frac, "inf_fraction": inf_frac,
        "zero_fraction": zero_frac, "clipping_fraction": clip,
        "baseline_drift": float(np.median(drift)) if drift else 0.0,
        "hf_ratio": float(np.median(hf)) if hf else 0.0,
        "lead_names": list(lead_names),
        "note": "hf_ratio is an auxiliary quality indicator, not a noise verdict",
    }


def summarize_source_signals(X: np.ndarray, meta, fs: float,
                             lead_names: list[str], max_records: int = 200):
    """Aggregate audit_signal results by source (bounded scan)."""
    import pandas as pd
    n = min(len(meta), max_records)
    step = max(1, n // max(1, 200))
    rows = []
    for i, offset in enumerate(range(0, n, step)):
        r = audit_signal(np.asarray(X[offset]), fs, lead_names)
        r["source"] = str(meta.source.iloc[offset])
        r["record_id"] = meta.record_id.iloc[offset]
        rows.append(r)
    df = pd.DataFrame(rows)
    return df.groupby("source").agg(
        n=("source", "size"), mean_rms=("rms", "mean"),
        mean_std=("std", "mean"), mean_zero=("zero_fraction", "mean"),
        mean_baseline=("baseline_drift", "mean"), mean_hf=("hf_ratio", "mean"),
    ).reset_index()
