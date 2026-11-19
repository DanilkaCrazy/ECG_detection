"""Stage 3.6 P0 — RED tests for corrected spectral / temporal / coherence metrics."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ecgdet.atrial_signal_analysis import (
    spectral_atrial, temporal_atrial, multilead_coherence, qrs_cancel_compare,
)

FS = 500
N = FS * 10
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def sines(freqs, amps, noise=0.002, seed=1):
    t = np.arange(N) / FS
    x = np.zeros(N)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return x + np.random.default_rng(seed).normal(0, noise, N)


def test_peak_width_is_contiguous_around_dominant_peak():
    # Two well-separated spectral components: dominant 4 Hz, weaker 8 Hz.
    x = sines([4.0, 8.0], [1.0, 0.35])
    f = spectral_atrial(x, FS)
    assert f["peak_width"] < 0.35, f  # must NOT span from 4 to 8 Hz


def test_harmonic_ratio_uses_specific_2f_and_3f_bands():
    # Dominant 4 Hz plus a 2nd harmonic at 8 Hz with known amplitude ratio 0.4.
    x = sines([4.0, 8.0], [1.0, 0.4])
    f = spectral_atrial(x, FS)
    assert "harmonic_2f_ratio" in f and "harmonic_3f_ratio" in f
    # power ratio: (0.4 amplitude)^2 = 0.16, with spectral leakage allowance
    assert 0.10 < f["harmonic_2f_ratio"] < 0.35, f
    assert f["harmonic_3f_ratio"] < 0.10, f


def test_peak_area_ratios_follow_hoppe_hypothesis():
    rng = np.random.default_rng(3)
    narrow = sines([5.0], [1.0], noise=0.05)
    broadband = rng.normal(0, 1.0, N)
    a = spectral_atrial(narrow, FS)
    b = spectral_atrial(broadband, FS)
    for key in ("peak_area_ratio_0_625hz", "peak_area_ratio_1_25hz",
                "peak_area_ratio_2_5hz", "peak_area_ratio_3_75hz",
                "peak_area_ratio_5hz", "peak_sharpness_db",
                "spectral_peak_isolation", "spectral_flatness"):
        assert key in a, key
    assert a["peak_area_ratio_2_5hz"] > b["peak_area_ratio_2_5hz"], (a, b)
    assert a["spectral_flatness"] < b["spectral_flatness"]


def test_temporal_uses_atrial_peaks_not_qrs_peaks():
    # Atrial cycles at 4.5 Hz (~222 ms); QRS peaks at 800 ms must be irrelevant.
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * 4.5 * t)
    atrial_peaks = np.arange(int(0.05 * FS), N - 1, int(FS / 4.5))
    qrs_peaks = np.arange(int(0.4 * FS), N - 1, int(0.8 * FS))
    f = temporal_atrial(x, FS, atrial_peaks, qrs_peaks=qrs_peaks)
    assert abs(f["atrial_cycle_mean"] - 222.2) < 15, f
    assert "rr_mean" in f and abs(f["rr_mean"] - 800.0) < 15, f


def test_periodicity_covers_full_3_to_10_hz_range():
    # 10 Hz atrial activity corresponds to a 100 ms lag (50 samples).
    x = sines([10.0], [1.0])
    f = temporal_atrial(x, FS, np.array([]))
    assert f["periodicity"] > 0.5, f


def test_multilead_coherence_is_spectral_not_pearson():
    t = np.arange(N) / FS
    a = np.sin(2 * np.pi * 5 * t)
    b = np.cos(2 * np.pi * 5 * t)          # 90 deg phase shift -> Pearson ~ 0
    c = np.random.default_rng(9).normal(0, 1, N)
    sig = np.asarray([a, b, c])
    f = multilead_coherence(sig, FS, ["II", "III", "V1"])
    assert "cross_spectral_coherence_mean" in f
    assert "phase_locking_value_mean" in f
    assert f["cross_spectral_coherence_mean"] > f["cross_lead_coherence_mean"], f
    assert f["phase_locking_value_mean"] > 0.3, f


def test_preservation_measured_inside_qrs_window():
    # Atrial sine that exists ONLY inside the QRS windows: preservation must
    # see it there; the old outside-mask metric would report ~1 regardless.
    t = np.arange(N) / FS
    qrs_peaks = np.arange(int(0.4 * FS), N - 1, int(0.8 * FS))
    mask = np.zeros(N, dtype=bool)
    for p in qrs_peaks:
        mask[max(0, p - 40):p + 60] = True
    atrial = np.sin(2 * np.pi * 4.5 * t) * mask
    qrs = np.zeros(N)
    for p in qrs_peaks:
        qrs[p - 15:p + 15] += np.hanning(30) * 3.0
    sig = np.asarray([qrs + atrial, qrs + atrial])
    res = qrs_cancel_compare(sig, FS, ["II", "V1"], qrs_peaks)
    for method in ("median", "adaptive", "beat_aligned"):
        assert "atrial_preservation_inside_qrs" in res[method], res[method]
        assert 0.0 <= res[method]["atrial_preservation_inside_qrs"] <= 1.5
