"""Stage 3.6.1 RED tests — semantics of residual quality metrics and masks."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.atrial_residual import extract_atrial_residual

FS = 500
LEADS = ["II", "V1"]


def _sig():
    rng = np.random.default_rng(0)
    n = FS * 10
    t = np.arange(n) / FS
    qrs = np.zeros(n)
    L = 40
    tmpl = np.hanning(L) * np.sin(np.linspace(0, 2 * np.pi, L))
    peaks = []
    for k in range(1, 13):
        c = int(k * 0.8 * FS)
        qrs[c - L // 2:c - L // 2 + L] += tmpl
        peaks.append(c)
    atrial = 0.15 * np.sin(2 * np.pi * 4.5 * t)
    return (qrs + atrial + rng.normal(0, 0.005, n))[None, :], np.asarray(peaks), n


def test_metric_is_named_residual_energy_ratio_not_preservation():
    sig, peaks, n = _sig()
    r = extract_atrial_residual(sig, FS, ["II"], method="median", qrs_peaks=peaks)
    assert "residual_energy_ratio_inside_qrs" in r
    assert "atrial_preservation_inside_qrs" not in r
    assert "atrial_preservation_quality" not in r


def test_tq_has_no_fake_suppression_and_own_metrics():
    sig, peaks, n = _sig()
    r = extract_atrial_residual(sig, FS, ["II"], method="tq_segment", qrs_peaks=peaks)
    assert np.isnan(r["qrs_suppression_quality"])
    for key in ("valid_tq_duration", "valid_tq_fraction", "n_valid_tq_segments",
                "median_tq_segment_length", "tq_extraction_status"):
        assert key in r, key
    assert r["tq_extraction_status"] in ("ok", "degraded", "failed")


def test_explicit_masks_have_distinct_meaning():
    sig, peaks, n = _sig()
    sub = extract_atrial_residual(sig, FS, ["II"], method="median", qrs_peaks=peaks)
    tq = extract_atrial_residual(sig, FS, ["II"], method="tq_segment", qrs_peaks=peaks)
    for r in (sub, tq):
        assert "qrs_exclusion_mask" in r and "valid_sample_mask" in r
        assert "qrs_mask" not in r
    assert sub["qrs_exclusion_mask"].sum() > 0
    assert sub["valid_sample_mask"].sum() > tq["valid_sample_mask"].sum()
    assert np.allclose(sub["qrs_exclusion_mask"] & tq["valid_sample_mask"], 0)
