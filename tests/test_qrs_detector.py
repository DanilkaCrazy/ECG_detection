"""Stage 3.6 P0 — regression tests for sparse/artifact-robust QRS detection."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.atrial_signal_analysis import multi_lead_qrs

FS = 500
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _sig(noise=0.01, artifact=0.0):
    """12 QRS complexes at 0.8 s; noise=0 mimics near-zero-baseline PTB-XL."""
    n = FS * 10
    rng = np.random.default_rng(0)
    sig = rng.normal(0, noise, (12, n))
    L = int(0.08 * FS)
    tmpl = np.hanning(L) * np.sin(np.linspace(0, 2 * np.pi, L))
    for k in range(1, 13):
        c = int(k * 0.8 * FS)
        for li in (1, 6, 10):  # II, V1, V5
            sig[li, c - L // 2:c - L // 2 + L] += tmpl
    if artifact:
        for li in (1, 6, 10):
            sig[li, 1000:1010] += artifact
    return sig


def test_qrs_detection_with_noise():
    peaks, _ = multi_lead_qrs(_sig(noise=0.01), FS, LEADS)
    assert len(peaks) >= 8, len(peaks)


def test_qrs_detection_on_sparse_near_zero_baseline():
    # PTB-XL-like: band energy ~0 outside QRS -> mean+std threshold fails
    peaks, _ = multi_lead_qrs(_sig(noise=0.0), FS, LEADS)
    assert len(peaks) >= 8, f"sparse baseline suppressed detection: {len(peaks)}"


def test_qrs_detection_robust_to_large_artifact():
    peaks, _ = multi_lead_qrs(_sig(noise=0.001, artifact=50.0), FS, LEADS)
    assert len(peaks) >= 8, f"artifact suppressed detection: {len(peaks)}"


def test_transposed_input_is_auto_detected():
    peaks, _ = multi_lead_qrs(_sig(noise=0.01).T, FS, LEADS)
    assert len(peaks) >= 8, len(peaks)
