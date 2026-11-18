"""Extended AF/AFL features: multi-lead RR analysis, Poincaré metrics,
sample entropy and flutter-wave activity in inferior leads.

Built on top of ``features_advanced.extract_advanced_features``.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as shannon_entropy

from .features_advanced import extract_advanced_features

__all__ = ["extract_v2_features"]


def _rr_series(x: np.ndarray, fs: float) -> np.ndarray:
    try:
        b, a = sp_signal.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        xf = sp_signal.filtfilt(b, a, x.astype(np.float64))
    except Exception:
        xf = x
    thr = np.mean(np.abs(xf)) + 1.5 * np.std(xf)
    try:
        peaks, _ = sp_signal.find_peaks(xf, height=thr, distance=int(0.2 * fs),
                                        prominence=max(thr * 0.3, 1e-9))
    except Exception:
        return np.array([])
    rr = np.diff(peaks) / fs * 1000
    return rr[(rr > 300) & (rr < 2000)]


def _sampen(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 10:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0:
        return np.nan
    pats = [x[i:i + m] for i in range(len(x) - m + 1)]
    c = []
    for i, p in enumerate(pats):
        c.append(sum(1 for j, q in enumerate(pats) if i != j and np.max(np.abs(p - q)) <= r))
    c = np.array(c, dtype=float)
    return float(np.log((c.sum() + 1e-12) / (len(pats) * max(len(pats) - 1, 1)))) if c.sum() > 0 else np.nan


def extract_v2_features(sig2d: np.ndarray, fs: float = 500.0) -> dict:
    sig2d = np.asarray(sig2d, dtype=np.float64)
    if sig2d.ndim == 1:
        sig2d = sig2d.reshape(-1, 1)
    f = extract_advanced_features(sig2d, fs=fs)

    n_leads = sig2d.shape[1]
    for name, idx in (("ii", 1 if n_leads > 1 else 0),
                      ("v1", 6 if n_leads > 6 else 0),
                      ("v5", 10 if n_leads > 10 else 0)):
        rr = _rr_series(sig2d[:, idx], fs)
        if len(rr) >= 3:
            d = np.diff(rr)
            f[f"v2_{name}_rr_std"] = float(np.std(rr))
            f[f"v2_{name}_rr_cv"] = float(np.std(rr) / (np.mean(rr) + 1e-9))
            f[f"v2_{name}_rr_rmssd"] = float(np.sqrt(np.mean(d ** 2)))
            f[f"v2_{name}_rr_pnn50"] = float(np.mean(np.abs(d) > 50) * 100)
            # Poincaré SD1/SD2
            f[f"v2_{name}_sd1"] = float(np.sqrt(0.5 * np.mean(d ** 2)))
            f[f"v2_{name}_sd2"] = float(np.sqrt(2 * np.var(rr)))
            f[f"v2_{name}_rr_sampen"] = _sampen(rr)
            f[f"v2_{name}_n_rr"] = float(len(rr))
            # tachogram spectral power (requires interpolation)
            try:
                t = np.cumsum(rr)
                t = t - t[0]
                grid = np.arange(0, t[-1], 50.0)  # 20 Hz
                if len(grid) > 4:
                    interp = np.interp(grid, t, rr)
                    fr, psd = sp_signal.welch(interp, fs=20.0, nperseg=min(256, len(interp)))
                    tot = psd.sum() + 1e-12
                    f[f"v2_{name}_rr_lf"] = float(psd[(fr >= 0.04) & (fr < 0.15)].sum() / tot)
                    f[f"v2_{name}_rr_hf"] = float(psd[(fr >= 0.15) & (fr < 0.4)].sum() / tot)
                    f[f"v2_{name}_rr_lfhf"] = float(f[f"v2_{name}_rr_lf"] / (f[f"v2_{name}_rr_hf"] + 1e-12))
            except Exception:
                pass
        else:
            for k in ("rr_std", "rr_cv", "rr_rmssd", "rr_pnn50", "sd1", "sd2",
                      "rr_sampen", "n_rr", "rr_lf", "rr_hf", "rr_lfhf"):
                f[f"v2_{name}_{k}"] = np.nan

    # flutter activity in II, III, aVF, V1 — maximal autocorr peak
    acs = []
    for idx in (1, 2, 5, 6):
        if idx >= n_leads:
            continue
        x = sig2d[:, idx]
        try:
            b, a = sp_signal.butter(3, [3 / (fs / 2), 12 / (fs / 2)], btype="band")
            xf = sp_signal.filtfilt(b, a, x)
        except Exception:
            continue
        xf = xf - xf.mean()
        if np.std(xf) < 1e-9:
            continue
        n = len(xf)
        ac = sp_signal.correlate(xf, xf, mode="full")[n - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(0.15 * fs), min(int(0.35 * fs), len(ac) - 1)
        if hi > lo:
            acs.append(float(np.max(ac[lo:hi])))
    if acs:
        f["v2_flutter_ac_max"] = float(np.max(acs))
        f["v2_flutter_ac_mean"] = float(np.mean(acs))
        f["v2_flutter_ac_std"] = float(np.std(acs))

    # NOTE: sample entropy on the raw 600-sample signal is O(n^2) in Python and
    # dominated the runtime; RR-level sample entropy above is cheap and kept.
    return f
