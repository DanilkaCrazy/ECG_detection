"""Stage 3.6.5 — atrial rhythm organization descriptors (P1)."""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal

__all__ = ["atrial_organization_features"]


def atrial_organization_features(x, fs, lo=3.0, hi=12.0):
    """Organization of atrial activity: periodicity, phase, morphology, spectral."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    out = {}
    xb = x - np.nanmean(x)
    if len(xb) < 64:
        return {k: np.nan for k in ("periodicity", "autocorr_half_life", "phase_locking",
                                    "amplitude_cv", "cycle_morph_consistency",
                                    "spectral_concentration", "instantaneous_freq_cv")}
    # autocorrelation periodicity in atrial cycle range
    ac = sp_signal.correlate(xb, xb, mode="full")[len(xb) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag_lo, lag_hi = max(1, int(fs / hi)), min(int(fs / lo), len(ac) - 1)
    out["periodicity"] = float(np.max(ac[lag_lo:lag_hi])) if lag_hi > lag_lo else np.nan
    half = np.where(ac[1:] < 0.5)[0]
    out["autocorr_half_life"] = float(half[0] + 1) / fs if len(half) else np.nan

    # Hilbert phase consistency and amplitude variability
    analytic = sp_signal.hilbert(xb)
    phase = np.angle(analytic)
    out["phase_locking"] = float(np.abs(np.mean(np.exp(1j * phase))))
    amp = np.abs(analytic)
    out["amplitude_cv"] = float(np.std(amp) / (np.mean(amp) + 1e-9))

    # instantaneous frequency variability in atrial band
    inst = np.diff(np.unwrap(phase)) * fs / (2 * np.pi)
    good = (inst >= lo) & (inst <= hi)
    out["instantaneous_freq_cv"] = float(np.std(inst[good]) / (np.mean(inst[good]) + 1e-9)) \
        if good.sum() > 10 else np.nan

    # cycle-to-cycle morphology consistency
    f, p = sp_signal.welch(xb, fs=fs, nperseg=min(1024, len(xb)))
    m = (f >= lo) & (f <= hi)
    total = p[m].sum() + 1e-12
    out["spectral_concentration"] = float(np.max(p[m]) / total) if m.any() else np.nan

    # F-wave cycles from bandpassed signal peaks
    bp = xb.copy()
    b, a = sp_signal.butter(3, [lo / (fs / 2), min(hi / (fs / 2), 0.99)], btype="band")
    bp = sp_signal.filtfilt(b, a, bp)
    pk, _ = sp_signal.find_peaks(bp, distance=max(1, int(fs / hi)),
                                 prominence=max(np.std(bp) * 0.3, 1e-9))
    L = max(8, int(fs / 8))
    cycles = []
    for q in pk:
        s, e = q - L // 2, q + L // 2
        if s >= 0 and e <= len(bp):
            c = bp[s:e]
            cycles.append(c / (np.std(c) + 1e-9))
    if len(cycles) >= 2:
        C = np.asarray(cycles)
        corr = np.nan_to_num(np.corrcoef(C))
        iu = np.triu_indices(len(C), 1)
        out["cycle_morph_consistency"] = float(np.mean(corr[iu]))
    else:
        out["cycle_morph_consistency"] = np.nan
    return out
