"""Stage 3.6.5 — time-frequency representations of atrial activity.

STFT / CWT / FSWT descriptors computed on the QRST-residual signal supplied by
the caller. These are hypothesis-generating features; validate on synthetic
ground truth before clinical interpretation.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
import pywt

__all__ = ["stft_atrial_features", "cwt_atrial_features", "fswt_atrial_features",
           "time_frequency_features"]


def _band_stats(freqs, times, power, lo=3.0, hi=12.0):
    """Common descriptors from a (freq, time) power map."""
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any() or power.shape[0] != len(freqs):
        return {}
    P = power[m]
    fp = freqs[m]
    dom = np.argmax(P, axis=0)
    dom_freq = fp[dom]
    total = P.sum(axis=0) + 1e-12
    prob = P / total
    entropy = -np.sum(prob * np.log(prob + 1e-12), axis=0)
    ridge_entropy = -np.sum((prob / (prob.sum(axis=1, keepdims=True) + 1e-12)) *
                            np.log(prob / (prob.sum(axis=1, keepdims=True) + 1e-12) + 1e-12))
    flux = float(np.mean(np.abs(np.diff(prob, axis=1))))
    return {
        "dominant_freq_mean": float(np.mean(dom_freq)),
        "dominant_freq_std": float(np.std(dom_freq)),
        "dominant_freq_cv": float(np.std(dom_freq) / (np.mean(dom_freq) + 1e-9)),
        "tf_entropy_mean": float(np.mean(entropy)),
        "tf_entropy_std": float(np.std(entropy)),
        "spectral_flux": flux,
        "band_energy": float(np.mean(total)),
        "band_energy_cv": float(np.std(total) / (np.mean(total) + 1e-9)),
        "scale_entropy": float(ridge_entropy),
    }


def stft_atrial_features(x, fs, lo=3.0, hi=12.0, nperseg=512):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if len(x) < nperseg:
        return {k: np.nan for k in ("dominant_freq_mean", "dominant_freq_std", "dominant_freq_cv",
                                    "tf_entropy_mean", "tf_entropy_std", "spectral_flux",
                                    "band_energy", "band_energy_cv", "scale_entropy")}
    f, t, Z = sp_signal.stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return _band_stats(f, t, np.abs(Z) ** 2, lo, hi)


def cwt_atrial_features(x, fs, lo=3.0, hi=12.0, wavelet="cmor1.5-1.0", n_scales=48):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    freqs_probe = np.linspace(0.5, 60, 200)
    central = pywt.central_frequency(wavelet)  # normalized cycles/sample
    scales_probe = central * fs / freqs_probe
    valid = (freqs_probe >= lo) & (freqs_probe <= hi)
    scales = np.linspace(scales_probe[valid].min(), scales_probe[valid].max(), n_scales) \
        if valid.any() else np.linspace(1, 100, n_scales)
    coef, f = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    power = np.abs(coef) ** 2
    # ridge frequency per time sample
    ridge = f[np.argmax(power, axis=0)]
    out = _band_stats(f, np.arange(power.shape[1]), power, lo, hi)
    out.update({"ridge_freq_mean": float(np.mean(ridge)),
                "ridge_freq_std": float(np.std(ridge)),
                "ridge_freq_cv": float(np.std(ridge) / (np.mean(ridge) + 1e-9))})
    return out


def fswt_atrial_features(x, fs, lo=3.0, hi=12.0, step=0.5, bandwidth=1.0):
    """Frequency-Slice Wavelet Transform via Gaussian slices in FFT domain."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    n = len(x)
    if n < 64:
        return {}
    freqs = np.arange(lo, hi + 1e-9, step)
    Xf = np.fft.rfft(x)
    ff = np.fft.rfftfreq(n, 1.0 / fs)
    P = []
    for fc in freqs:
        w = np.exp(-((ff - fc) ** 2) / (2 * bandwidth ** 2))
        comp = np.fft.irfft(Xf * w, n=n)
        analytic = sp_signal.hilbert(comp)
        P.append(np.abs(analytic) ** 2)
    power = np.asarray(P)
    out = _band_stats(freqs, np.arange(n), power, lo, hi)
    out.update({"ridge_freq_mean": float(freqs[np.argmax(power.mean(axis=1))]),
                "ridge_freq_std": float(np.std(freqs[
                    np.argmax(power, axis=0)]))})
    return out


def time_frequency_features(x, fs):
    out = {}
    for name, fn in (("stft", stft_atrial_features), ("cwt", cwt_atrial_features),
                     ("fswt", fswt_atrial_features)):
        for k, v in fn(x, fs).items():
            out[f"{name}_{k}"] = v
    return out
