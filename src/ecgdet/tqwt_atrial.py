"""Stage 3.6.5 (P1) — TQWT-inspired atrial/QRST decomposition.

NOTE: this is a *TQWT-inspired* wavelet-packet decomposition with Q-controlled
subband grouping, not an exact reproduction of Zhu et al. (2022). It is provided
as an independent experimental method to compare against template subtraction
and TQ segments. Validate on synthetic signals with known components before
interpretation on clinical ECG.
"""
from __future__ import annotations
import numpy as np
import pywt

__all__ = ["tqwt_atrial_decompose", "decomposition_quality"]


def _band_for_path(path: str, fs: float):
    """Approximate frequency band [lo, hi] of a wavelet-packet node."""
    lo, hi = 0.0, fs / 2.0
    for ch in path:
        mid = (lo + hi) / 2.0
        if ch == "d":
            lo = mid
        else:
            hi = mid
    return lo, hi


def tqwt_atrial_decompose(x, fs, *, level=6, atrial_band=(3.0, 12.0), wavelet="db4"):
    """Return (atrial_component, qrst_component, subband_info)."""
    x = np.asarray(x, dtype=float)
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    atrial = np.zeros_like(x)
    qrst = np.zeros_like(x)
    info = []
    lo_a, hi_a = atrial_band
    paths = [n.path for n in nodes]
    for node in nodes:
        band = _band_for_path(node.path, fs)
        # Reconstruct a single subband on a fresh tree: zero all other leaves.
        tmp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=level)
        for p in paths:
            if p != node.path:
                tmp[p].data = np.zeros_like(tmp[p].data)
        comp = tmp.reconstruct()
        if band[1] > lo_a and band[0] < hi_a:
            atrial += comp
            role = "atrial"
        else:
            qrst += comp
            role = "qrst"
        info.append({"path": node.path, "band": band, "role": role})
    return atrial, qrst, info


def decomposition_quality(x, fs, atrial_hat, qrst_hat, atrial_true, qrs_true):
    """Objective metrics on synthetic signals with known ground truth."""
    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    def snr(sig, err):
        return float(10 * np.log10((np.mean(np.asarray(sig) ** 2) + 1e-12) /
                                   (np.mean(np.asarray(err) ** 2) + 1e-12)))
    corr_atrial = float(np.corrcoef(atrial_hat, atrial_true)[0, 1])
    corr_qrs = float(np.corrcoef(qrst_hat, qrs_true)[0, 1])
    # Reconstruction quality: both components together must rebuild x.
    resid = np.asarray(x) - np.asarray(qrst_hat) - np.asarray(atrial_hat)
    return {
        "atrial_rmse": rmse(atrial_hat, atrial_true),
        "atrial_correlation": corr_atrial,
        "qrs_correlation": corr_qrs,
        "reconstruction_error_ratio": float(np.mean(resid ** 2) /
                                            (np.mean(np.asarray(x) ** 2) + 1e-12)),
        "atrial_snr_db": snr(atrial_true, atrial_hat - atrial_true),
    }
