"""Advanced ECG feature extraction aimed at AF vs AFL discrimination.

On top of the base features (``ecgdet.features.extract_features``) this adds:
- per-lead spectral band powers and ratios (flutter activity ~4-7 Hz)
- spectral entropy, dominant frequency, peakiness
- R-peak / RR statistics: irregularity (RMSSD, pNN50, CV, entropy), autocorr
- atrial activity proxies: autocorrelation peak at flutter-cycle lags (0.15-0.35 s)
- QRS-peak morphology: width/energy variability across beats
- cross-lead correlation statistics
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as shannon_entropy

from .features import LEAD_ORDER_12, extract_features

__all__ = ["extract_advanced_features", "ADVANCED_FEATURE_GROUPS"]

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 15.0),
    ("beta", 15.0, 30.0),
    ("gamma", 30.0, 60.0),
    ("hf", 60.0, 100.0),
]


def _safe(x) -> float:
    x = float(x)
    return x if np.isfinite(x) else np.nan


def _spectral_features(x: np.ndarray, fs: float) -> dict:
    out: dict[str, float] = {}
    n = len(x)
    if n < 64:
        return out
    nperseg = min(512, n)
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    total = float(np.sum(psd)) + 1e-12
    band_powers: dict[str, float] = {}
    for name, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        p = float(np.sum(psd[m]))
        band_powers[name] = p
        out[f"psd_{name}_abs"] = _safe(p)
        out[f"psd_{name}_rel"] = _safe(p / total)
    out["psd_theta_alpha_ratio"] = _safe(band_powers["theta"] / (band_powers["alpha"] + 1e-12))
    out["psd_low_high_ratio"] = _safe(
        (band_powers["delta"] + band_powers["theta"] + band_powers["alpha"])
        / (band_powers["beta"] + band_powers["gamma"] + band_powers["hf"] + 1e-12)
    )
    prob = psd / total
    out["psd_entropy"] = _safe(shannon_entropy(prob))
    m_pos = freqs > 0
    out["psd_centroid"] = _safe(float(np.sum(freqs[m_pos] * psd[m_pos]) / (np.sum(psd[m_pos]) + 1e-12)))
    # dominant frequency in the atrial-activity range
    m_atrial = (freqs >= 3.0) & (freqs <= 12.0)
    if m_atrial.any():
        idx = int(np.argmax(psd[m_atrial]))
        out["atrial_dom_freq"] = _safe(freqs[m_atrial][idx])
        out["atrial_dom_power"] = _safe(psd[m_atrial][idx] / total)
    else:
        out["atrial_dom_freq"] = np.nan
        out["atrial_dom_power"] = np.nan
    return out


def _peak_rr_features(sig2d: np.ndarray, fs: float, lead_idx: int) -> dict:
    out: dict[str, float] = {}
    x = sig2d[:, lead_idx].astype(np.float64)
    if len(x) < fs:
        return out
    try:
        b, a = sp_signal.butter(3, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype="band")
        xf = sp_signal.filtfilt(b, a, x)
    except Exception:
        xf = x
    thr = np.mean(np.abs(xf)) + 1.5 * np.std(xf)
    try:
        peaks, props = sp_signal.find_peaks(
            xf, height=thr, distance=int(0.2 * fs), prominence=max(thr * 0.3, 1e-9),
            width=(1, int(0.2 * fs)),
        )
    except Exception:
        peaks = np.array([], dtype=int)
        props = {}
    out["n_rpeaks"] = _safe(len(peaks))
    out["rpeak_height_mean"] = _safe(np.mean(props.get("peak_heights", np.zeros(len(peaks)))) if len(peaks) else np.nan)
    out["rpeak_height_cv"] = _safe(
        np.std(props.get("peak_heights", np.zeros(len(peaks)))) / (np.mean(props.get("peak_heights", [1])) + 1e-9)
        if len(peaks) else np.nan
    )
    w = props.get("widths", np.zeros(len(peaks)))
    out["rpeak_width_mean_ms"] = _safe(np.mean(w) / fs * 1000 if len(w) else np.nan)
    out["rpeak_width_cv"] = _safe(np.std(w) / (np.mean(w) + 1e-9) if len(w) else np.nan)

    if len(peaks) < 3:
        for k in ("hr_mean", "rr_mean", "rr_std", "rr_cv", "rr_rmssd", "rr_pnn50",
                  "rr_pnn20", "rr_entropy", "rr_ac1", "rr_iqr", "rr_range"):
            out[k] = np.nan
        return out
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 2:
        for k in ("rr_std", "rr_cv", "rr_rmssd", "rr_pnn50", "rr_pnn20", "rr_entropy", "rr_ac1", "rr_iqr", "rr_range"):
            out[k] = np.nan
        out["hr_mean"] = _safe(60000.0 / np.mean(np.diff(peaks) / fs))
        out["rr_mean"] = _safe(np.mean(rr)) if len(rr) else np.nan
        return out
    d = np.diff(rr)
    out["hr_mean"] = _safe(60000.0 / np.mean(rr))
    out["rr_mean"] = _safe(np.mean(rr))
    out["rr_std"] = _safe(np.std(rr))
    out["rr_cv"] = _safe(np.std(rr) / (np.mean(rr) + 1e-9))
    out["rr_rmssd"] = _safe(np.sqrt(np.mean(d ** 2)))
    out["rr_pnn50"] = _safe(np.mean(np.abs(d) > 50) * 100.0)
    out["rr_pnn20"] = _safe(np.mean(np.abs(d) > 20) * 100.0)
    out["rr_iqr"] = _safe(np.percentile(rr, 75) - np.percentile(rr, 25))
    out["rr_range"] = _safe(np.max(rr) - np.min(rr))
    hist, _ = np.histogram(rr, bins=min(10, max(3, len(rr) // 3)))
    out["rr_entropy"] = _safe(shannon_entropy(hist + 1e-9))
    out["rr_ac1"] = _safe(np.corrcoef(rr[:-1], rr[1:])[0, 1]) if len(rr) > 3 else np.nan
    return out


def _atrial_activity(x: np.ndarray, fs: float) -> dict:
    """Detect regular 4-7 Hz atrial activity (flutter waves) via autocorrelation."""
    out: dict[str, float] = {}
    n = len(x)
    if n < fs:
        return out
    try:
        b, a = sp_signal.butter(3, [3.0 / (fs / 2), 12.0 / (fs / 2)], btype="band")
        xf = sp_signal.filtfilt(b, a, x.astype(np.float64))
    except Exception:
        return out
    xf = xf - np.mean(xf)
    if np.std(xf) < 1e-9:
        return out
    ac = sp_signal.correlate(xf, xf, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.15 * fs), int(0.35 * fs)  # 170-400 bpm atrial cycle
    lo, hi = max(1, lo), min(len(ac) - 1, hi)
    if hi > lo:
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        out["flutter_ac_peak"] = _safe(seg[k])
        out["flutter_period_s"] = _safe((lo + k) / fs)
        out["flutter_rate_bpm"] = _safe(60000.0 / (1000.0 * (lo + k) / fs))
    # spectral peakiness in flutter band
    freqs, psd = sp_signal.welch(xf, fs=fs, nperseg=min(1024, n))
    m = (freqs >= 3.5) & (freqs <= 7.0)
    if m.any():
        out["flutter_peak_ratio"] = _safe(np.max(psd[m]) / (np.mean(psd[m]) + 1e-12))
    return out


def extract_advanced_features(sig2d: np.ndarray, fs: float = 500.0) -> dict:
    """Extract base + advanced features from (n_samples, n_leads) ECG array."""
    sig2d = np.asarray(sig2d, dtype=np.float64)
    if sig2d.ndim == 1:
        sig2d = sig2d.reshape(-1, 1)
    feats = extract_features(sig2d, fs=fs, lead_names=LEAD_ORDER_12[: sig2d.shape[1]])

    # spectral features per lead
    for i in range(sig2d.shape[1]):
        for k, v in _spectral_features(sig2d[:, i], fs).items():
            feats[f"spec{i}_{k}"] = v

    # peak/RR features from lead II
    lead_ii = 1 if sig2d.shape[1] >= 2 else 0
    for k, v in _peak_rr_features(sig2d, fs, lead_ii).items():
        feats[f"rr_{k}"] = v

    # atrial activity in leads II and V1 (index 6)
    for name, idx in (("ii", lead_ii), ("v1", 6 if sig2d.shape[1] > 6 else 0)):
        for k, v in _atrial_activity(sig2d[:, idx], fs).items():
            feats[f"atrial_{name}_{k}"] = v

    # cross-lead correlation
    if sig2d.shape[1] >= 2:
        corr = np.corrcoef(sig2d.T)
        corr = np.nan_to_num(corr)
        iu = np.triu_indices(corr.shape[0], k=1)
        vals = corr[iu]
        feats["cross_corr_mean"] = _safe(np.mean(vals))
        feats["cross_corr_std"] = _safe(np.std(vals))
        feats["cross_corr_absmax"] = _safe(np.max(np.abs(vals)))
    return feats
