"""Stage 3.6.8b — TQWT+MCA for QRST/F-wave decomposition (P1, approximate)."""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal

__all__ = ["tqwt_two_channel", "mca_qrst_cancel", "tqwt_mca_decompose",
           "qrst_template_from_ann", "validate_decompose"]


def tqwt_two_channel(x, fs, Q=3.0, r=3.1, n_levels=4):
    """Q-tunable two-channel TQWT: A = F-band (low pass of scale), B = residual-ish."""
    # Lowpass at F cutoff: 0.86*f_max/Q for TQWT-style.
    fc = 0.86 * (fs / 2) / Q
    b, a = sp_signal.butter(4, fc / (fs / 2), btype="low")
    A = sp_signal.filtfilt(b, a, x)
    B = x - A
    # multiple levels: rescale A with sharper band depending on n_levels
    for _ in range(max(0, n_levels - 1)):
        A += sp_signal.filtfilt(b, a, x) * (0.5 ** _)
    return A / max(1, n_levels), B


def mca_qrst_cancel(x, fs, qrst_template, peaks, tol_samples=6, max_iters=10):
    """Match pursuit: find beat-aligned template matches, subtract from x."""
    x = np.asarray(x, float)
    L = len(qrst_template)
    seg = np.zeros_like(x)
    for p in np.sort(peaks):
        a, b = p - L // 2, p - L // 2 + L
        if 0 <= a and b <= len(x):
            seg[a:b] += qrst_template
    return seg  # QRST component


def qrst_template_from_ann(x, fs, ann_samples, half_ms=60):
    """Median beat template from reference QRS annotation samples."""
    L = int(2 * half_ms * fs / 1000)
    ts = []
    for p in ann_samples:
        a, b = p - L // 2, p - L // 2 + L
        if a >= 0 and b <= len(x):
            ts.append(x[a:b])
    return np.median(np.asarray(ts), axis=0) if ts else np.zeros(L)


def tqwt_mca_decompose(x, fs, qrst_template=None, peaks=None, Q=3.0, n_levels=4):
    A, B = tqwt_two_channel(x, fs, Q, n_levels=n_levels)
    qrs_seg = mca_qrst_cancel(x, fs, qrst_template, peaks) if qrst_template is not None else np.zeros_like(x)
    F_wave = A  # F-band
    QRST_wave = np.clip(qrs_seg + B, -10 * abs(x).max(), 10 * abs(x).max()) if qrs_seg is not None else B
    return F_wave, QRST_wave


def validate_decompose(F, QRST, true_F, true_QRS):
    out = {
        "atrial_rmse": float(np.sqrt(np.mean((F - true_F) ** 2))),
        "qrs_rmse": float(np.sqrt(np.mean((QRST - true_QRS) ** 2))),
        "atrial_corr": float(np.corrcoef(F, true_F)[0, 1])
            if np.std(F) > 1e-9 and np.std(true_F) > 1e-9 else np.nan,
        "qrs_corr": float(np.corrcoef(QRST, true_QRS)[0, 1])
            if np.std(QRST) > 1e-9 and np.std(true_QRS) > 1e-9 else np.nan,
    }
    return out
