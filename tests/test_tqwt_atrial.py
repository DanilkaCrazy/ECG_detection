"""Stage 3.6.5 — TQWT-inspired decomposition on synthetic ground truth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.tqwt_atrial import tqwt_atrial_decompose, decomposition_quality

FS = 500


def synth():
    n = FS * 10
    t = np.arange(n) / FS
    qrs = np.zeros(n)
    L = 40
    tmpl = np.hanning(L) * np.sin(np.linspace(0, 2 * np.pi, L))
    for k in range(1, 13):
        c = int(k * 0.8 * FS)
        qrs[c - L // 2:c - L // 2 + L] += tmpl
    atrial = 0.15 * np.sin(2 * np.pi * 5 * t)
    x = qrs + atrial + np.random.default_rng(0).normal(0, 0.003, n)
    return x, atrial, qrs


def test_decomposition_recovers_known_atrial_and_qrs():
    x, atrial, qrs = synth()
    a_hat, q_hat, info = tqwt_atrial_decompose(x, FS, level=6)
    assert len(info) == 64
    q = decomposition_quality(x, FS, a_hat, q_hat, atrial, qrs)
    assert q["atrial_correlation"] > 0.5, q
    # Frequency-band-only separation is approximate: part of QRS lies in 3-12 Hz.
    # Full morphological MCA is not implemented yet.
    assert q["qrs_correlation"] > 0.4, q
    assert q["reconstruction_error_ratio"] < 0.05, q
