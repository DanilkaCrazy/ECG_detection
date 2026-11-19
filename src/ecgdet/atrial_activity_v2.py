"""Stage 3.4 — physiologically motivated atrial-activity analysis.

Pipeline per lead (focus II, III, aVF, V1):
1. R-peak detection (bandpass + adaptive threshold);
2. QRST cancellation via per-beat template subtraction with interpolation;
3. residual (atrial-only) signal QC — cancellation quality is reported;
4. frequency features: dominant frequency, spectral entropy/concentration,
   atrial cycle estimate, dominant-frequency stability across subwindows;
5. temporal features: cycle length stats, autocorrelation peak, periodicity score;
6. morphology: cycle-to-cycle correlation, amplitude variability, sawtooth score;
7. multilead: cross-lead correlation and spatial consistency.

IMPORTANT: cancellation quality is reported and poor cancellations are flagged;
features from unreliable residuals must not be trusted silently.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

__all__ = ["ATRIAL_LEADS", "detect_r_peaks", "qr_cancel", "atrial_features"]

ATRIAL_LEADS = {"II": 1, "III": 2, "aVF": 5, "V1": 6}


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < fs // 2:
        return np.array([], dtype=int)
    try:
        b, a = sp_signal.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        xf = sp_signal.filtfilt(b, a, x)
    except Exception:
        xf = x
    thr = np.mean(np.abs(xf)) + 1.5 * np.std(xf)
    peaks, _ = sp_signal.find_peaks(xf, height=thr, distance=int(0.2 * fs),
                                    prominence=max(thr * 0.3, 1e-9))
    return peaks.astype(int)


def qr_cancel(x: np.ndarray, fs: float, peaks: np.ndarray | None = None,
              half_pre: float = 0.08, half_post: float = 0.12):
    """Remove ventricular complexes by subtracting an averaged beat template.

    Returns (residual, removed_mask, quality_dict). Quality includes the
    fraction of signal removed and input NaN fraction.
    """
    x = np.asarray(x, dtype=np.float64)
    q = {"input_nan_frac": float(np.isnan(x).mean()),
         "input_inf_frac": float(np.isinf(x).mean())}
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if peaks is None:
        peaks = detect_r_peaks(x, fs)
    mask = np.zeros(len(x), dtype=bool)
    pre, post = int(half_pre * fs), int(half_post * fs)
    for p in peaks:
        lo, hi = max(0, p - pre), min(len(x), p + post)
        mask[lo:hi] = True
    # build average template from all beats (aligned to R peak)
    L = pre + post
    beats = []
    for p in peaks:
        lo, hi = p - pre, p + post
        if lo >= 0 and hi <= len(x):
            beats.append(x[lo:hi])
    residual = x.copy()
    if len(beats) >= 2:
        template = np.median(np.asarray(beats), axis=0)
        # taper template edges to avoid discontinuities
        w = np.hanning(L)
        for p in peaks:
            lo, hi = p - pre, p + post
            if lo >= 0 and hi <= len(x):
                residual[lo:hi] -= template * w
    residual -= np.mean(residual)
    q["qrs_removed_ratio"] = float(mask.mean())
    q["n_beats"] = int(len(peaks))
    q["residual_std"] = float(np.std(residual))
    q["cancellation_reliable"] = bool(len(beats) >= 2 and 0.01 < mask.mean() < 0.6)
    return residual, mask, q


def _spectral(x, fs):
    out = {}
    if len(x) < 128:
        return {"atr_dom_freq": np.nan, "atr_spec_entropy": np.nan,
                "atr_spec_concentration": np.nan}
    nperseg = min(1024, len(x))
    f, p = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    band = (f >= 3) & (f <= 12)
    if band.any():
        pb = p[band]
        j = int(np.argmax(pb))
        out["atr_dom_freq"] = float(f[band][j])
        tot = pb.sum() + 1e-12
        out["atr_spec_concentration"] = float(pb[j] / tot)
        pr = pb / tot
        out["atr_spec_entropy"] = float(-np.sum(pr * np.log(pr + 1e-12)))
    else:
        out.update({"atr_dom_freq": np.nan, "atr_spec_entropy": np.nan,
                    "atr_spec_concentration": np.nan})
    # stability of dominant frequency across 3 subwindows
    dom = []
    for part in np.array_split(x, 3):
        if len(part) < 128:
            continue
        fp, pp = sp_signal.welch(part, fs=fs, nperseg=min(512, len(part)))
        m = (fp >= 3) & (fp <= 12)
        if m.any():
            dom.append(float(fp[m][int(np.argmax(pp[m]))]))
    out["atr_dom_freq_stability"] = float(np.std(dom)) if len(dom) >= 2 else np.nan
    return out


def _temporal(residual, fs):
    out = {}
    peaks = detect_r_peaks(residual, fs)
    if len(peaks) >= 3:
        cyc = np.diff(peaks) / fs * 1000.0
        cyc = cyc[(cyc > 150) & (cyc < 400)]
        if len(cyc) >= 2:
            out["atr_cycle_mean"] = float(np.mean(cyc))
            out["atr_cycle_median"] = float(np.median(cyc))
            out["atr_cycle_std"] = float(np.std(cyc))
            out["atr_cycle_cv"] = float(np.std(cyc) / (np.mean(cyc) + 1e-9))
        else:
            out.update({"atr_cycle_mean": np.nan, "atr_cycle_median": np.nan,
                        "atr_cycle_std": np.nan, "atr_cycle_cv": np.nan})
    else:
        out.update({"atr_cycle_mean": np.nan, "atr_cycle_median": np.nan,
                    "atr_cycle_std": np.nan, "atr_cycle_cv": np.nan})
    # autocorrelation peak in atrial cycle range (171-400 ms)
    if len(residual) > 512:
        r = residual - residual.mean()
        ac = sp_signal.correlate(r, r, mode="full")[len(r)-1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(0.171 * fs), min(int(0.4 * fs), len(ac)-1)
        if hi > lo:
            out["atr_ac_peak"] = float(np.max(ac[lo:hi]))
        else:
            out["atr_ac_peak"] = np.nan
    else:
        out["atr_ac_peak"] = np.nan
    out["atr_periodicity_score"] = out["atr_ac_peak"]
    return out


def _morphology(residual, fs, peaks):
    out = {}
    if peaks is None or len(peaks) < 3:
        return {"atr_morph_consistency": np.nan, "atr_amp_variability": np.nan,
                "atr_sawtooth_score": np.nan}
    L = int(0.2 * fs)
    cycles = []
    for p in peaks:
        lo, hi = p - L // 4, p + L - L // 4
        if lo >= 0 and hi <= len(residual):
            c = residual[lo:hi]
            cycles.append(c / (np.std(c) + 1e-9))
    if len(cycles) >= 2:
        C = np.asarray(cycles)
        corr = np.corrcoef(C)
        iu = np.triu_indices(len(C), 1)
        out["atr_morph_consistency"] = float(np.nanmean(corr[iu]))
        amps = [float(np.std(residual[max(0, p-L//4):min(len(residual), p+L-L//4)]))
                for p in peaks]
        out["atr_amp_variability"] = float(np.std(amps) / (np.mean(amps) + 1e-9))
        # sawtooth-like: asymmetric rise/fall of averaged cycle
        avg = np.mean(C, axis=0)
        half = len(avg) // 2
        out["atr_sawtooth_score"] = float(np.mean(avg[:half]) - np.mean(avg[half:]))
    else:
        out.update({"atr_morph_consistency": np.nan, "atr_amp_variability": np.nan,
                    "atr_sawtooth_score": np.nan})
    return out


def atrial_features(sig2d: np.ndarray, fs: float = 500.0,
                    lead_names: list[str] | None = None) -> dict:
    """Compute Stage 3.4 atrial features for a (n_leads, n_samples) signal."""
    sig2d = np.asarray(sig2d, dtype=np.float64)
    names = [str(x) for x in lead_names] if lead_names is not None else \
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    out: dict[str, float] = {}
    residuals = {}
    for lead, idx in ATRIAL_LEADS.items():
        if idx >= sig2d.shape[0]:
            continue
        r, mask, q = qr_cancel(sig2d[idx], fs)
        residuals[lead] = r
        for k, v in q.items():
            out[f"atr_{lead}_{k}"] = float(v)
        # de-trend and band-pass atrial band
        x = sp_signal.detrend(r)
        try:
            b, a = sp_signal.butter(3, [3 / (fs / 2), 12 / (fs / 2)], btype="band")
            x = sp_signal.filtfilt(b, a, x)
        except Exception:
            pass
        for k, v in _spectral(x, fs).items():
            out[f"{k}_{lead}"] = v
        peaks = detect_r_peaks(r, fs)
        for k, v in _temporal(x, fs).items():
            out[f"{k}_{lead}"] = v
        for k, v in _morphology(x, fs, peaks).items():
            out[f"{k}_{lead}"] = v
    # multilead consistency
    if len(residuals) >= 2:
        mats = np.asarray([r - r.mean() for r in residuals.values()])
        corr = np.corrcoef(mats)
        iu = np.triu_indices(len(mats), 1)
        out["atr_crosslead_corr_mean"] = float(np.nanmean(corr[iu]))
        out["atr_crosslead_corr_std"] = float(np.nanstd(corr[iu]))
        freqs = [out.get("atr_dom_freq_" + l, np.nan) for l in residuals]
        freqs = np.array([f for f in freqs if np.isfinite(f)])
        out["atr_spatial_freq_consistency"] = float(np.std(freqs) / (np.mean(freqs)+1e-9)) if len(freqs) > 1 else np.nan
        per = {k: np.array([out.get(f"{k}_{lead}", np.nan) for lead in residuals], dtype=float)
               for k in ("atr_periodicity_score", "atr_morph_consistency",
                         "atr_sawtooth_score", "atr_dom_freq", "atr_spec_concentration")}
        out["atrial_periodicity_score"] = float(np.nanmean(per["atr_periodicity_score"]))
        out["atrial_morphology_consistency"] = float(np.nanmean(per["atr_morph_consistency"]))
        out["multilead_fwave_consistency"] = float(np.nanmean(per["atr_sawtooth_score"]))
        out["atrial_rate_global"] = float(np.nanmean(per["atr_dom_freq"]) * 60.0)
        out["atrial_spectral_concentration"] = float(np.nanmean(per["atr_spec_concentration"]))
    else:
        out["atr_crosslead_corr_mean"] = np.nan
        out["atr_crosslead_corr_std"] = np.nan
        out["atr_spatial_freq_consistency"] = np.nan
    return out
