"""Stage 3.6.9 — lead availability and signal-quality audit (P0).

A missing/constant lead must be represented by an explicit mask and passed to
downstream modules.  It must never be silently interpreted as a physiologically
meaningful zero signal, and a record must not be dropped just because several
leads are unavailable.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

__all__ = [
    "CANONICAL_LEADS",
    "lead_quality",
    "build_lead_mask",
    "summarize_dataset_leads",
]

CANONICAL_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)

_ALIASES = {
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "avr ": "aVR", "ii": "II", "v1": "V1",
}


def _canonical(name: Any) -> str:
    s = str(name).strip()
    return _ALIASES.get(s.lower(), s)


def lead_quality(x: np.ndarray, fs: float, lead_names: Sequence[str]) -> pd.DataFrame:
    """Per-lead quality statistics for a (leads, samples) signal.

    ``usable`` is False when the lead is constant/zero, has excessive NaN/Inf,
    or is dominated by high-frequency noise.  Short stretches of NaN are
    tolerated: they are imputed downstream by the mask-aware feature code.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signal must be (leads, samples)")
    if x.shape[0] != len(lead_names):
        raise ValueError("lead_names length must match signal rows")
    fs = float(fs)
    rows = []
    for i, raw_name in enumerate(lead_names):
        name = _canonical(raw_name)
        lead = x[i]
        n = len(lead)
        finite = np.isfinite(lead)
        nan_frac = float(np.mean(np.isnan(lead))) if n else 1.0
        inf_frac = float(np.mean(np.isinf(lead))) if n else 0.0
        clean = lead[finite]
        if clean.size == 0:
            rows.append({"lead": name, "zero_frac": 1.0, "nan_frac": nan_frac,
                         "inf_frac": inf_frac, "amplitude": 0.0, "noise_rms": 0.0,
                         "usable": False})
            continue
        zero_frac = float(np.mean(np.abs(clean) < 1e-8))
        amplitude = float(np.percentile(np.abs(clean), 95))
        # High-frequency noise estimate: energy above 40 Hz relative to total.
        if clean.size >= 32 and fs > 80:
            freqs = np.fft.rfftfreq(clean.size, 1.0 / fs)
            spec = np.abs(np.fft.rfft(clean - clean.mean()))
            total = float(np.sum(spec ** 2))
            hf = float(np.sum(spec[freqs > 40.0] ** 2)) if total > 0 else 0.0
            hf_ratio = hf / total if total > 0 else 0.0
            noise_rms = float(np.sqrt(hf / max(n, 1))) if hf > 0 else 0.0
        else:
            hf_ratio, noise_rms = 0.0, 0.0
        usable = bool(
            zero_frac < 0.9
            and nan_frac < 0.2
            and inf_frac < 0.05
            and amplitude > 1e-4
            and hf_ratio < 0.9
        )
        rows.append({"lead": name, "zero_frac": zero_frac, "nan_frac": nan_frac,
                     "inf_frac": inf_frac, "amplitude": amplitude,
                     "noise_rms": noise_rms, "usable": usable})
    return pd.DataFrame(rows)


def build_lead_mask(x: np.ndarray, fs: float, lead_names: Sequence[str]) -> np.ndarray:
    """Boolean availability mask aligned with ``lead_names``."""
    return lead_quality(x, fs, lead_names)["usable"].to_numpy(dtype=bool)


def summarize_dataset_leads(
    X: np.ndarray,
    meta: pd.DataFrame,
    fs: float,
    lead_names: Sequence[str] = CANONICAL_LEADS,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Aggregate lead usability per source (bounded scan for large datasets)."""
    X = np.asarray(X)
    meta = meta.reset_index(drop=True)
    n = len(meta) if max_records is None else min(len(meta), max_records)
    step = max(1, n // max(1, 200))
    positions = np.arange(0, n, step)
    sampled = meta.iloc[positions].reset_index(drop=True)
    masks = [build_lead_mask(X[p], fs, lead_names) for p in positions]
    arr = np.asarray(masks, dtype=bool)
    rows = []
    for src in sorted(sampled["source"].astype(str).unique()):
        idx = np.where(sampled["source"].astype(str).to_numpy() == src)[0]
        sub = arr[idx] if len(idx) else arr[:0]
        rows.append({
            "source": str(src),
            "n_sampled": int(len(idx)),
            "mean_usable_leads": float(sub.mean()) if sub.size else float("nan"),
            "min_usable_leads": int(sub.sum(axis=1).min()) if sub.size else 0,
            "all_leads_usable_frac": float(np.mean(sub.all(axis=1))) if sub.size else float("nan"),
            "per_lead_usable": {
                _canonical(name): float(sub[:, k].mean()) if sub.size else float("nan")
                for k, name in enumerate(lead_names)
            },
        })
    return pd.DataFrame(rows)
