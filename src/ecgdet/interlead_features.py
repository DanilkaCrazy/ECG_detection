"""Stage 3.6.10 — Group E: genuine inter-lead synchronisation features.

Unlike the legacy ``phase_locking`` descriptor (which measures concentration
of ONE signal's instantaneous phase), every quantity here requires TWO leads:

* magnitude-squared coherence in the atrial band 3-10 Hz (cross-spectrum);
* phase-locking value of the band-passed analytic signals;
* cross-lead amplitude consistency and dominant-frequency agreement.

An explicit ``lead_mask`` is honoured, so an unavailable lead never becomes a
physiologically meaningful all-zero signal.  Computation is vectorised over
all lead pairs so the full 12-lead montage stays cheap at dataset scale.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

__all__ = ["interlead_features", "FEATURE_NAMES"]

FEATURE_NAMES = (
    "interlead_coherence_mean", "interlead_coherence_std",
    "interlead_plv_mean", "interlead_plv_std",
    "interlead_amp_cv", "interlead_domfreq_spread", "n_leads_used",
)

DEFAULT_LEADS = ("II", "III", "aVF", "V1")


def _bandpass(x: np.ndarray, fs: float, lo: float = 3.0, hi: float = 10.0) -> np.ndarray:
    nyq = fs / 2.0
    if hi >= nyq or lo <= 0 or len(x) < 16:
        return np.asarray(x, dtype=float)
    b, a = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, np.asarray(x, dtype=float))


def interlead_features(
    x: np.ndarray,
    fs: float,
    lead_names: list[str],
    lead_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute cross-lead coherence/PLV on all available leads."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("signal must be (leads, samples)")
    if x.shape[0] != len(lead_names):
        raise ValueError("lead_names length must match signal rows")
    names = [str(n).strip() for n in lead_names]
    idx = {n: i for i, n in enumerate(names)}
    mask = np.ones(len(names), dtype=bool) if lead_mask is None else np.asarray(lead_mask, bool)

    available = [i for i in range(len(names)) if mask[i]]
    ordered = [idx[n] for n in DEFAULT_LEADS if n in idx and mask[idx[n]]]
    picks = ordered + [i for i in available if i not in ordered]
    if len(picks) < 2:
        return {k: 0.0 for k in FEATURE_NAMES} | {"n_leads_used": float(len(picks))}

    fs = float(fs)
    filtered = np.asarray([_bandpass(x[i], fs) for i in picks])
    n = filtered.shape[1]
    spectrum = np.fft.rfft(filtered, axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs >= 3.0) & (freqs <= 10.0)
    if not band.any():
        band = np.ones(len(freqs), dtype=bool)

    s = spectrum[:, band]
    power = np.abs(s) ** 2
    cross = s[:, :, None] * s.conj()[:, None, :]  # (leads, leads, freq)
    iu = np.triu_indices(len(picks), k=1)
    denom = power[:, None, :] * power[None, :, :]
    coh = np.abs(cross[iu[0], iu[1], :]) ** 2 / np.maximum(
        denom[iu[0], iu[1], :], 1e-12)
    coherences = np.mean(coh, axis=1)

    analytic = np.asarray([sp_signal.hilbert(sig) for sig in filtered])
    phase = np.angle(analytic)
    delta = phase[:, None, :] - phase[None, :, :]
    plvs = np.abs(np.mean(np.exp(1j * delta[iu]), axis=1))

    amps = np.mean(np.abs(analytic), axis=1)
    mean_amp = float(amps.mean())
    amp_cv = float(amps.std() / mean_amp) if mean_amp > 1e-12 else 0.0
    domfreqs = np.array([freqs[np.argmax(np.abs(spectrum[k]) ** 2)] for k in range(len(picks))])

    return {
        "interlead_coherence_mean": float(np.mean(coherences)),
        "interlead_coherence_std": float(np.std(coherences)),
        "interlead_plv_mean": float(np.mean(plvs)),
        "interlead_plv_std": float(np.std(plvs)),
        "interlead_amp_cv": amp_cv,
        "interlead_domfreq_spread": float(np.std(domfreqs)),
        "n_leads_used": float(len(picks)),
    }
