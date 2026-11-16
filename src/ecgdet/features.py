"""Feature extraction for classical ML from ECG.

Focus on robust, interpretable features:
- Time-domain statistics per lead + cross-lead
- RR intervals / HRV (if enough length)
- Morphological summaries (min/max/mean/median/iqr/skew/kurtosis)
- Zero-crossing rate, energy, peak-to-peak
- Hjorth parameters (activity/mobility/complexity)
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

__all__ = [
    "extract_features",
    "lead_names_standard12",
]


LEAD_ORDER_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# common aliases
ALIAS = {
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
}


def lead_names_standard12() -> list[str]:
    return list(LEAD_ORDER_12)


def _hjorth(x: np.ndarray):
    x = np.asarray(x).ravel()
    if len(x) < 3:
        return 0.0, 0.0, 0.0
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(ddx)
    if var0 == 0:
        return 0.0, 0.0, 0.0
    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(var2 / var1) / mobility if var1 > 0 and mobility > 0 else 0
    return activity, mobility, complexity


def _lead_features(x: np.ndarray) -> dict:
    x = np.asarray(x).ravel()
    feats = {}
    feats["mean"] = float(np.mean(x))
    feats["median"] = float(np.median(x))
    feats["std"] = float(np.std(x))
    feats["var"] = float(np.var(x))
    feats["min"] = float(np.min(x))
    feats["max"] = float(np.max(x))
    feats["ptp"] = float(np.ptp(x))
    feats["iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    feats["skew"] = float(stats.skew(x))
    feats["kurtosis"] = float(stats.kurtosis(x))
    feats["mad"] = float(np.median(np.abs(x - np.median(x))))
    feats["rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats["abs_mean"] = float(np.mean(np.abs(x)))
    feats["energy"] = float(np.sum(x ** 2))
    # zero crossings
    signs = np.sign(x)
    signs[signs == 0] = 1
    feats["zcr"] = float(np.sum(np.diff(signs) != 0) / len(x))
    # Hjorth
    a, m, c = _hjorth(x)
    feats["hjorth_activity"] = float(a)
    feats["hjorth_mobility"] = float(m)
    feats["hjorth_complexity"] = float(c)
    # quantiles
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        feats[f"q{q}"] = float(np.percentile(x, q))
    return feats


def _rr_features(sig2d: np.ndarray, fs: float) -> dict:
    """Naive R-peak detection via Pan-Tompkins-like heuristic on lead II if present."""
    feats = {}
    if sig2d.ndim != 2:
        return feats
    # pick lead II
    lead_names = []
    if sig2d.shape[1] == 12:
        lead_names = LEAD_ORDER_12
    else:
        lead_names = [str(i) for i in range(sig2d.shape[1])]
    idx = None
    for i, ln in enumerate(lead_names):
        if ln.upper() in ("II", "2"):
            idx = i
            break
    if idx is None and sig2d.shape[1] >= 2:
        idx = 1
    if idx is None:
        return feats
    x = sig2d[:, idx]
    # bandpass-ish smoothing
    try:
        b, a = signal.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        xf = signal.filtfilt(b, a, x)
    except Exception:
        xf = x
    # find peaks
    try:
        from scipy.signal import find_peaks
        thr = np.mean(np.abs(xf)) + 1.5 * np.std(xf)
        peaks, _ = find_peaks(xf, height=thr, distance=int(0.2 * fs), prominence=thr * 0.3)
    except Exception:
        peaks = []
    if len(peaks) < 3:
        feats["hr_mean"] = np.nan
        feats["rr_std"] = np.nan
        feats["rr_rmssd"] = np.nan
        feats["rr_pnn50"] = np.nan
        feats["n_peaks"] = int(len(peaks))
        return feats
    rr = np.diff(peaks) / fs * 1000  # ms
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 2:
        feats["hr_mean"] = 60000.0 / np.mean(np.diff(peaks) / fs) if len(peaks) > 1 else np.nan
        feats["rr_std"] = np.nan
        feats["rr_rmssd"] = np.nan
        feats["rr_pnn50"] = np.nan
        feats["n_peaks"] = int(len(peaks))
        return feats
    feats["hr_mean"] = float(60000.0 / np.mean(rr))
    feats["rr_mean"] = float(np.mean(rr))
    feats["rr_std"] = float(np.std(rr))
    feats["rr_median"] = float(np.median(rr))
    diff = np.diff(rr)
    feats["rr_rmssd"] = float(np.sqrt(np.mean(diff ** 2)))
    feats["rr_pnn50"] = float(np.mean(np.abs(diff) > 50) * 100.0)
    feats["rr_min"] = float(np.min(rr))
    feats["rr_max"] = float(np.max(rr))
    feats["n_peaks"] = int(len(peaks))
    return feats


def extract_features(sig2d: np.ndarray, fs: float, lead_names: list[str] | None = None) -> dict:
    """Extract a flat dict of features from a 2D ECG array (n_samples, n_leads)."""
    sig2d = np.asarray(sig2d, dtype=np.float64)
    if sig2d.ndim == 1:
        sig2d = sig2d.reshape(-1, 1)
    if lead_names is None:
        if sig2d.shape[1] == 12:
            lead_names = LEAD_ORDER_12
        else:
            lead_names = [f"lead{i}" for i in range(sig2d.shape[1])]
    feats = {}
    for i, ln in enumerate(lead_names):
        if i >= sig2d.shape[1]:
            break
        for k, v in _lead_features(sig2d[:, i]).items():
            feats[f"{ln}_{k}"] = v
    # cross-lead
    if sig2d.shape[1] >= 2:
        feats["global_mean"] = float(np.mean(sig2d))
        feats["global_std"] = float(np.std(sig2d))
        feats["global_ptp"] = float(np.ptp(sig2d))
    feats.update(_rr_features(sig2d, fs))
    return feats
