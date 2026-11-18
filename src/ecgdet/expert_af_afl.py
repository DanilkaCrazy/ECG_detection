"""Physiologically-motivated AF/AFL expert features.

Implements the feature groups proposed for AF vs AFL discrimination:
A. RR interval statistics and nonlinear entropy measures
B. Spectral characteristics of atrial activity
C. Wavelet-like multi-resolution band energies (filter-bank approximation of DWT)
D. Atrial wave morphology / organisation
E. AV conduction: atrial-to-ventricular rate ratio and conduction regularity

All features are computed from a (n_samples, n_leads) ECG array in mV at fs Hz.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as shannon_entropy

__all__ = ["extract_expert_features", "EXPERT_GROUPS"]

EXPERT_GROUPS = ["A_rr", "B_spectral", "C_wavelet", "D_atrial", "E_av"]


def _finite(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _bandpower(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    if len(x) < 64:
        return np.nan
    nperseg = min(1024, len(x))
    f, p = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    m = (f >= lo) & (f < hi)
    return float(np.sum(p[m])) if m.any() else 0.0


def _sampen(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample entropy (cheap enough for RR sequences of length ~10-30)."""
    x = np.asarray(x, dtype=float)
    if len(x) < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0:
        return np.nan
    pats = np.lib.stride_tricks.sliding_window_view(x, m)
    c = []
    for i, p in enumerate(pats):
        c.append(int(np.sum(np.max(np.abs(pats - p), axis=1) <= r)) - 1)
    c = np.array(c, dtype=float)
    s1 = np.sum(c[c > 0])
    if s1 <= 0:
        return np.nan
    pats2 = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    c2 = []
    for i, p in enumerate(pats2):
        c2.append(int(np.sum(np.max(np.abs(pats2 - p), axis=1) <= r)) - 1)
    s2 = np.sum(np.array(c2)[np.array(c2) > 0])
    return float(-np.log((s2 + 1e-12) / (s1 + 1e-12)))


def _approx_entropy(x: np.ndarray, m: int = 2) -> float:
    """Approximate entropy (simplified, for short RR series)."""
    x = np.asarray(x, dtype=float)
    if len(x) < m + 2:
        return np.nan
    r = 0.2 * np.std(x)
    if r <= 0:
        return np.nan
    def phi(mm: int) -> float:
        pats = np.lib.stride_tricks.sliding_window_view(x, mm)
        counts = np.array([np.sum(np.max(np.abs(pats - p), axis=1) <= r) for p in pats])
        return float(np.mean(np.log(np.maximum(counts, 1) / len(x))))
    return _finite(phi(m) - phi(m + 1))


def _detect_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    try:
        b, a = sp_signal.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        xf = sp_signal.filtfilt(b, a, x.astype(float))
    except Exception:
        xf = x
    thr = np.mean(np.abs(xf)) + 1.5 * np.std(xf)
    try:
        peaks, _ = sp_signal.find_peaks(xf, height=thr, distance=int(0.2 * fs),
                                        prominence=max(thr * 0.3, 1e-9))
    except Exception:
        return np.array([], dtype=int)
    return peaks


def extract_expert_features(sig2d: np.ndarray, fs: float = 500.0) -> dict:
    sig2d = np.asarray(sig2d, dtype=np.float64)
    if sig2d.ndim == 1:
        sig2d = sig2d.reshape(-1, 1)
    nlead = sig2d.shape[1]
    out: dict[str, float] = {}

    lead_ii = 1 if nlead > 1 else 0
    xii = sig2d[:, lead_ii]

    # ---- Group A: RR intervals ----
    peaks = _detect_peaks(xii, fs)
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) >= 3:
        d = np.diff(rr)
        out["A_rr_mean"] = _finite(np.mean(rr))
        out["A_rr_median"] = _finite(np.median(rr))
        out["A_sdnn"] = _finite(np.std(rr))
        out["A_rmssd"] = _finite(np.sqrt(np.mean(d ** 2)))
        out["A_cvr"] = _finite(np.std(rr) / (np.mean(rr) + 1e-9))
        out["A_pnn50"] = _finite(np.mean(np.abs(d) > 50) * 100)
        out["A_hr"] = _finite(60000 / np.mean(rr))
        h, _ = np.histogram(rr, bins=min(8, max(3, len(rr))))
        out["A_rr_shannon"] = _finite(shannon_entropy(h + 1e-9))
        out["A_rr_sampen"] = _sampen(rr)
        out["A_rr_approxen"] = _approx_entropy(rr)
        out["A_rr_iqr"] = _finite(np.percentile(rr, 75) - np.percentile(rr, 25))
        out["A_rr_cv2"] = _finite(np.std(np.diff(rr)) / (np.mean(rr) + 1e-9))
        out["A_rr_sd1"] = _finite(np.sqrt(0.5 * np.mean(d ** 2)))
        out["A_rr_sd2"] = _finite(np.sqrt(2 * np.var(rr)))
    else:
        for k in ("A_rr_mean", "A_rr_median", "A_sdnn", "A_rmssd", "A_cvr", "A_pnn50",
                  "A_hr", "A_rr_shannon", "A_rr_sampen", "A_rr_approxen", "A_rr_iqr",
                  "A_rr_cv2", "A_rr_sd1", "A_rr_sd2"):
            out[k] = np.nan
    out["A_n_beats"] = float(len(peaks))

    # ---- Group B/C: spectral + wavelet-like band energies ----
    bands = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 16),
             ("beta", 16, 32), ("gamma", 32, 64), ("high", 64, 128)]
    for lead_name, idx in (("ii", lead_ii), ("v1", 6 if nlead > 6 else 0)):
        x = sig2d[:, idx]
        tot = sum(_bandpower(x, fs, lo, hi) for _, lo, hi in bands) + 1e-12
        for name, lo, hi in bands:
            p = _bandpower(x, fs, lo, hi)
            out[f"B_{lead_name}_{name}_abs"] = _finite(p)
            out[f"B_{lead_name}_{name}_rel"] = _finite(p / tot)
            out[f"C_{lead_name}_{name}_log"] = _finite(np.log10(p + 1e-12))
        out[f"B_{lead_name}_theta_alpha"] = _finite(
            out[f"B_{lead_name}_theta_abs"] / (out[f"B_{lead_name}_alpha_abs"] + 1e-12))
        out[f"B_{lead_name}_spectral_entropy"] = _finite(shannon_entropy(
            np.array([out[f"B_{lead_name}_{n}_rel"] for n, _, _ in bands]) + 1e-12))

    # ---- Group D: atrial wave organisation (II, III, aVF, V1) ----
    atrial_rates = []
    acs = []
    for idx in [1, 2, 5, 6]:
        if idx >= nlead:
            continue
        x = sig2d[:, idx]
        try:
            b, a = sp_signal.butter(3, [3 / (fs / 2), 12 / (fs / 2)], btype="band")
            xf = sp_signal.filtfilt(b, a, x - np.mean(x))
        except Exception:
            continue
        if np.std(xf) < 1e-9:
            continue
        n = len(xf)
        ac = sp_signal.correlate(xf, xf, mode="full")[n - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(0.15 * fs), min(int(0.4 * fs), len(ac) - 1)
        if hi > lo:
            k = int(np.argmax(ac[lo:hi]))
            acs.append(_finite(ac[lo + k]))
            atrial_rates.append(60000 / ((lo + k) / fs * 1000))
    if acs:
        out["D_atrial_ac_max"] = float(np.max(acs))
        out["D_atrial_ac_mean"] = float(np.mean(acs))
        out["D_atrial_ac_std"] = float(np.std(acs))
        out["D_atrial_rate_med"] = float(np.median(atrial_rates))
        out["D_atrial_rate_std"] = float(np.std(atrial_rates))
        # regularity of atrial cycle across leads -> organised flutter vs chaotic AF
        out["D_atrial_rate_cv"] = _finite(np.std(atrial_rates) / np.mean(atrial_rates))
    else:
        for k in ("D_atrial_ac_max", "D_atrial_ac_mean", "D_atrial_ac_std",
                  "D_atrial_rate_med", "D_atrial_rate_std", "D_atrial_rate_cv"):
            out[k] = np.nan

    # ---- Group E: AV conduction ----
    if len(rr) >= 3 and atrial_rates:
        ar = np.median(atrial_rates)
        vr = 60000 / np.mean(rr)
        ratio = ar / (vr + 1e-12)
        out["E_av_ratio"] = _finite(ratio)
        out["E_av_ratio_nearest"] = float(np.clip(np.round(ratio / 0.5) * 0.5, 0.5, 8.0))
        # how well ventricular intervals are integer multiples of atrial cycle
        atrial_cycle = 60000 / (ar + 1e-12)
        mult = rr / (atrial_cycle + 1e-12)
        out["E_mult_mean"] = _finite(np.mean(mult))
        out["E_mult_std"] = _finite(np.std(mult))
        out["E_mult_int_dev"] = _finite(np.mean(np.abs(mult - np.round(mult))))
        out["E_cond_regularity"] = _finite(1 - np.std(rr) / (np.mean(rr) + 1e-12))
    else:
        for k in ("E_av_ratio", "E_av_ratio_nearest", "E_mult_mean", "E_mult_std",
                  "E_mult_int_dev", "E_cond_regularity"):
            out[k] = np.nan
    return out
