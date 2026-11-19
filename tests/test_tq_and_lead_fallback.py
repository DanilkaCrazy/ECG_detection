"""Stage 3.6.3 RED tests — TQ short-RR safety and lead-fallback QRS detection."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.atrial_residual import extract_atrial_residual
from ecgdet.atrial_signal_analysis import multi_lead_qrs

FS = 500


def _qrs_sig(rr_ms, n_leads=3, valid_leads=(0, 1, 2), n_beats=None):
    n = FS * 10
    t = np.arange(n) / FS
    if n_beats is None:
        n_beats = int(9.5 / (rr_ms / 1000.0))
    sig = np.zeros((n_leads, n))
    L = 40
    tmpl = np.hanning(L) * np.sin(np.linspace(0, 2 * np.pi, L))
    peaks = []
    for k in range(1, n_beats + 1):
        c = int(k * rr_ms / 1000 * FS)
        peaks.append(c)
        for li in valid_leads:
            sig[li, c - L // 2:c - L // 2 + L] += tmpl
    atrial = 0.1 * np.sin(2 * np.pi * 5 * t)
    for li in valid_leads:
        sig[li] += atrial
    for li in range(n_leads):
        if li not in valid_leads:
            sig[li] = 0.0
    return sig, np.asarray(peaks)


def test_tq_never_overlaps_qrs_at_short_rr():
    for rr in (250, 300, 350, 400):
        sig, peaks = _qrs_sig(rr)
        r = extract_atrial_residual(sig, FS, ["II", "V1", "V5"],
                                    method="tq_segment", qrs_peaks=peaks)
        overlap = r["qrs_exclusion_mask"] & r["valid_sample_mask"]
        assert overlap.sum() == 0, (rr, int(overlap.sum()))
        assert np.isnan(r["qrs_suppression_quality"])


def test_tq_marks_invalid_short_intervals_without_forcing_segments():
    sig, peaks = _qrs_sig(250)
    r = extract_atrial_residual(sig, FS, ["II", "V1", "V5"],
                                method="tq_segment", qrs_peaks=peaks)
    assert "n_invalid_short_tq" in r
    assert r["n_valid_tq_segments"] >= 0
    # no artificial 8-sample segments inside a QRS window
    assert r["valid_sample_mask"].sum() <= int(0.05 * FS * len(peaks))


def test_qrs_detection_falls_back_when_chest_leads_are_flat():
    # II is good; V1 and V5 are exactly zero ( Chapman missing-lead case )
    sig, peaks = _qrs_sig(800, n_leads=12, valid_leads=(1,))
    leads = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]
    found, info = multi_lead_qrs(sig, FS, leads)
    assert len(found) >= 8, (len(found), info)
    assert info["qrs_detection_status"] in ("ok", "degraded")
    assert "qrs_detection_confidence" in info
