"""Stage 3.5 RED tests — advanced atrial signal analysis."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.atrial_signal_analysis import (
    multi_lead_qrs, qrs_cancel_compare, spectral_atrial, temporal_atrial,
    lead_index_map, multilead_coherence,
)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500


def synth(neg_ii=False, n_beats=12):
    n = FS * 10
    t = np.arange(n) / FS
    sig = np.random.default_rng(1).normal(0, 0.01, (12, n))
    for k in range(1, n_beats):
        c = int(k * 0.8 * FS)
        w = np.hanning(31)
        sig[1, c-15:c+16] += w * (-1.0 if neg_ii else 1.0)
        sig[6, c-15:c+16] += w * 0.8
        sig[10, c-15:c+16] += w * 0.6
    for k in range(1, int(10 / 0.2)):
        c = int(k * 0.2 * FS)
        for li in (1, 2, 5, 6):
            sig[li, c:c+6] += np.hanning(6) * 0.15
    return sig


def test_lead_index_map_by_name():
    idx = lead_index_map(["AVR", "V1", "II", "I"],)
    assert idx["aVR"] == 0 and idx["V1"] == 1 and idx["II"] == 2 and idx["I"] == 3


def test_multilead_qrs_handles_negative_polarity():
    sig = synth(neg_ii=True)
    peaks, q = multi_lead_qrs(sig, FS, LEADS)
    assert len(peaks) >= 8, (len(peaks), q)
    assert "consensus_leads" in q


def test_qrs_cancel_methods_compare_quality():
    sig = synth()
    peaks, _ = multi_lead_qrs(sig, FS, LEADS)
    out = qrs_cancel_compare(sig, FS, LEADS, peaks)
    assert set(out.keys()) >= {"median", "adaptive", "beat_aligned"}
    for k, v in out.items():
        assert "qrs_residual_energy" in v, (k, v)
        assert "residual_energy_ratio_outside_qrs" in v


def test_spectral_features():
    sig = synth()
    f = spectral_atrial(sig[1], FS)
    for k in ("dominant_freq", "peak_prominence", "peak_to_background",
              "spectral_entropy", "spectral_bandwidth", "peak_width", "harmonic_ratio"):
        assert k in f, k


def test_temporal_and_coherence():
    sig = synth()
    peaks, _ = multi_lead_qrs(sig, FS, LEADS)
    f = temporal_atrial(sig[1], FS, peaks)
    assert "atrial_cycle_cv" in f and "periodicity" in f
    coh = multilead_coherence(sig, FS, LEADS)
    assert "cross_lead_coherence_mean" in coh
