"""Stage 3.0 — atrial activity analysis with QRS suppression.

Focus leads: II, III, aVF, V1. Removes ventricular complexes by interpolating
across QRS windows, then measures regularity / spectral organisation of the
remaining atrial signal (flutter waves vs fibrillatory baseline).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from .expert_af_afl import _detect_peaks

__all__ = ["suppress_qrs", "atrial_activity_features", "ATRIAL_LEADS"]

ATRIAL_LEADS = {"II": 1, "III": 2, "aVF": 5, "V1": 6}


def suppress_qrs(x: np.ndarray, fs: float, peaks: np.ndarray | None = None,
                 half_width_ms: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate over QRS complexes; returns (cleaned_signal, removed_mask)."""
    x = np.asarray(x, dtype=np.float64)
    if peaks is None:
        peaks = _detect_peaks(x, fs)
    mask = np.zeros(len(x), dtype=bool)
    half = int(half_width_ms / 1000.0 * fs)
    for p in peaks:
        lo, hi = max(0, int(p) - half), min(len(x), int(p) + half)
        mask[lo:hi] = True
    cleaned = x.copy()
    good = np.where(~mask)[0]
    if len(good) > 2:
        cleaned[mask] = np.interp(np.where(mask)[0], good, x[good])
    return cleaned, mask


def atrial_activity_features(sig2d: np.ndarray, fs: float = 500.0) -> dict:
    """Physiological atrial-activity features from II, III, aVF and V1."""
    sig2d = np.asarray(sig2d, dtype=np.float64)
    out: dict[str, float] = {}
    refs = []
    for name, idx in ATRIAL_LEADS.items():
        if idx >= sig2d.shape[1]:
            continue
        x = sig2d[:, idx]
        peaks = _detect_peaks(x, fs)
        xc, mask = suppress_qrs(x, fs, peaks)
        # detrend and band-limit to atrial activity band
        xc = sp_signal.detrend(xc)
        try:
            b, a = sp_signal.butter(3, [3 / (fs / 2), 15 / (fs / 2)], btype="band")
            xf = sp_signal.filtfilt(b, a, xc)
        except Exception:
            xf = xc
        sd = np.std(xf)
        out[f"atr_{name}_amp"] = float(sd)
        out[f"atr_{name}_qrs_frac"] = float(mask.mean())
        if sd < 1e-9 or len(xf) < 256:
            for k in ("dom_freq", "peak_ratio", "ac_peak", "ac_period_ms", "cycle_cv"):
                out[f"atr_{name}_{k}"] = np.nan
            continue
        f, p = sp_signal.welch(xf, fs=fs, nperseg=min(1024, len(xf)))
        band = (f >= 3) & (f <= 12)
        if band.any():
            j = int(np.argmax(p[band]))
            out[f"atr_{name}_dom_freq"] = float(f[band][j])
            out[f"atr_{name}_peak_ratio"] = float(p[band][j] / (np.mean(p[band]) + 1e-12))
        else:
            out[f"atr_{name}_dom_freq"] = np.nan
            out[f"atr_{name}_peak_ratio"] = np.nan
        n = len(xf)
        ac = sp_signal.correlate(xf, xf, mode="full")[n - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(0.15 * fs), min(int(0.4 * fs), len(ac) - 1)
        if hi > lo:
            seg = ac[lo:hi]
            k = int(np.argmax(seg))
            out[f"atr_{name}_ac_peak"] = float(seg[k])
            out[f"atr_{name}_ac_period_ms"] = float((lo + k) / fs * 1000)
        else:
            out[f"atr_{name}_ac_peak"] = np.nan
            out[f"atr_{name}_ac_period_ms"] = np.nan
        refs.append((name, out.get(f"atr_{name}_ac_period_ms", np.nan)))

    # inter-lead coherence of the dominant atrial cycle
    periods = np.array([v for _, v in refs if np.isfinite(v)])
    out["atr_cycle_cv"] = float(np.std(periods) / (np.mean(periods) + 1e-9)) if len(periods) > 1 else np.nan
    out["atr_cycle_n_leads"] = float(len(periods))
    return out
