"""Stage 3.4 RED tests for atrial_activity module (QRST cancellation + features)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.atrial_activity_v2 import (
    detect_r_peaks, qr_cancel, atrial_features, ATRIAL_LEADS,
)


def make_synthetic(fs=500, dur=10, atrial_rate=300, noise=0.0, irregular=False):
    n = int(fs * dur)
    t = np.arange(n) / fs
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]
    sig = np.zeros((n, 12))
    # ventricular QRS every ~0.8 s
    for k in range(1, int(dur / 0.8)):
        c = int(k * 0.8 * fs)
        sig[c-10:c+15, :] += (np.hanning(25) * 1.0)[:, None]
    # atrial waves at given rate (organized)
    period = 60.0 / atrial_rate
    for k in range(1, int(dur / period)):
        c = int(k * period * fs)
        for li in ATRIAL_LEADS.values():
            if c+5 <= n:
                sig[c:c+5, li] += np.hanning(5) * 0.15
    if noise:
        sig += np.random.default_rng(0).normal(0, noise, sig.shape)
    return sig.T, lead_names  # (12, n)


def test_detect_r_peaks_finds_qrs():
    sig, _ = make_synthetic()
    peaks = detect_r_peaks(sig[1], 500.0)
    assert len(peaks) >= 8, len(peaks)


def test_qr_cancel_removes_qrs():
    sig, _ = make_synthetic(noise=0.0)
    residual, mask, q = qr_cancel(sig[1], 500.0)
    assert residual.shape == sig[1].shape
    assert 0 < mask.mean() < 0.3, mask.mean()
    assert 0 <= q["qrs_removed_ratio"] <= 1


def test_atrial_features_shapes_and_periodicity():
    sig, names = make_synthetic(atrial_rate=300)
    f = atrial_features(sig, 500.0, names)
    assert len(f) > 40, len(f)
    assert "atrial_periodicity_score" in f
    assert np.isfinite(f["atrial_periodicity_score"])


def test_multilead_features():
    sig, names = make_synthetic(atrial_rate=300)
    f = atrial_features(sig, 500.0, names)
    for lead in ("II", "III", "aVF", "V1"):
        assert any(k.startswith(f"atr_{lead}_") for k in f), lead


def test_qr_cancel_with_nan_returns_quality_flag():
    sig, _ = make_synthetic()
    sig[1, 100] = np.nan
    _, _, q = qr_cancel(sig[1], 500.0)
    assert q["input_nan_frac"] > 0
