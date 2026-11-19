"""Stage 3.6 P0 — RED tests for correct atrial residual extraction.

Synthetic signals with a KNOWN atrial component (4.5 Hz sine) and a KNOWN QRS
component are used, so suppression and preservation can be measured, not assumed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from ecgdet.atrial_residual import extract_atrial_residual

FS = 500
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def make_signal(atrial_hz=4.5, atrial_amp=0.15, qrs_period_s=0.8, noise=0.005, seed=0):
    """Return (signal[1,N], qrs[N], atrial[N], qrs_peaks) with known components."""
    rng = np.random.default_rng(seed)
    n = FS * 10
    t = np.arange(n) / FS
    qrs = np.zeros(n)
    L = int(0.08 * FS)
    tmpl = np.hanning(L) * np.sin(np.linspace(0, 2 * np.pi, L))
    peaks = []
    for k in range(1, int(10 / qrs_period_s)):
        c = int(k * qrs_period_s * FS)
        lo = c - L // 2
        qrs[lo:lo + L] += tmpl
        peaks.append(c)
    atrial = atrial_amp * np.sin(2 * np.pi * atrial_hz * t)
    sig = qrs + atrial + rng.normal(0, noise, n)
    return sig[None, :], qrs, atrial, np.asarray(peaks), t


def test_extract_atrial_residual_contract():
    sig, _, _, peaks, _ = make_signal()
    r = extract_atrial_residual(sig, FS, ["II"], method="median", qrs_peaks=peaks)
    for key in ("residual", "qrs_mask", "qrs_suppression_quality",
                "atrial_preservation_quality", "valid_leads", "extraction_status"):
        assert key in r, key
    assert r["residual"].shape == sig.shape
    assert r["qrs_mask"].shape == (sig.shape[1],)
    assert r["extraction_status"] in ("ok", "degraded", "failed")


def test_residual_suppresses_qrs_but_preserves_atrial():
    sig, qrs, atrial, peaks, _ = make_signal()
    r = extract_atrial_residual(sig, FS, ["II"], method="median", qrs_peaks=peaks)
    mask = r["qrs_mask"]
    assert mask.sum() > 100, "QRS mask must cover the QRS windows"
    resid = r["residual"][0]
    pre = float(np.mean(sig[0][mask] ** 2))
    post = float(np.mean(resid[mask] ** 2))
    assert post < 0.35 * pre, (pre, post)  # QRS energy removed
    corr = float(np.corrcoef(resid[mask], atrial[mask])[0, 1])
    assert corr > 0.7, corr  # known 4.5 Hz atrial wave survived
    assert r["qrs_suppression_quality"] > 0.6
    assert r["atrial_preservation_quality"] > 0.6


@pytest.mark.parametrize("method", ["median", "adaptive", "beat_aligned", "tq_segment"])
def test_all_methods_return_usable_residual(method):
    sig, _, _, peaks, _ = make_signal()
    sig2 = np.repeat(sig, 2, axis=0)  # two leads for two lead names
    r = extract_atrial_residual(sig2, FS, ["II", "V1"], method=method, qrs_peaks=peaks)
    assert r["residual"].shape == (2, FS * 10)
    # TQ segments intentionally leave NaN in discarded intervals; usable samples
    # are those marked by qrs_mask. Other methods return fully finite residuals.
    m = r["qrs_mask"]
    assert m.sum() > FS, method
    assert np.isfinite(r["residual"][:, m]).all(), method
    assert r["extraction_status"] != "failed", (method, r["extraction_status"])


def test_tq_segment_residual_keeps_atrial_frequency():
    from ecgdet.atrial_signal_analysis import spectral_atrial
    sig, _, _, peaks, _ = make_signal()
    r = extract_atrial_residual(sig, FS, ["II"], method="tq_segment", qrs_peaks=peaks)
    f = spectral_atrial(r["residual"][0], FS)
    assert 4.2 <= f["dominant_freq"] <= 4.8, f


def test_unreliable_extraction_is_marked_not_zero_filled():
    flat = np.zeros((1, FS * 10))
    r = extract_atrial_residual(flat, FS, ["II"], method="median", qrs_peaks=np.array([]))
    assert r["extraction_status"] in ("degraded", "failed")
    assert np.isnan(r["qrs_suppression_quality"]) or np.isnan(r["atrial_preservation_quality"])
