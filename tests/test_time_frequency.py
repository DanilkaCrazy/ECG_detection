"""Stage 3.6.5 RED — time-frequency and atrial organization features."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.time_frequency import stft_atrial_features, cwt_atrial_features, fswt_atrial_features
from ecgdet.organization import atrial_organization_features

FS = 500


def chirp_atrial(f0=4.0, f1=8.0, amp=0.15, noise=0.005, seed=0):
    n = FS * 10
    t = np.arange(n) / FS
    phase = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * t / 10.0) / FS
    x = (amp + 0.02 * np.sin(2 * np.pi * 0.4 * t)) * np.sin(phase)
    return x + np.random.default_rng(seed).normal(0, noise, n)


def test_stft_tracks_time_varying_dominant_frequency():
    x = chirp_atrial()
    f = stft_atrial_features(x, FS)
    assert "dominant_freq_mean" in f and "dominant_freq_std" in f
    assert 3.0 <= f["dominant_freq_mean"] <= 9.0, f
    assert f["dominant_freq_std"] > 0.3, f  # frequency actually drifts


def test_cwt_and_fswt_return_comparable_descriptors():
    x = chirp_atrial()
    a = cwt_atrial_features(x, FS)
    b = fswt_atrial_features(x, FS)
    for d in (a, b):
        assert "ridge_freq_mean" in d and "ridge_freq_std" in d
        assert "scale_entropy" in d
        assert 3.0 <= d["ridge_freq_mean"] <= 9.0, d


def test_organization_distinguishes_regular_from_irregular():
    t = np.arange(FS * 10) / FS
    regular = np.sin(2 * np.pi * 5 * t) + np.random.default_rng(0).normal(0, 0.005, len(t))
    irregular = np.random.default_rng(1).normal(0, 0.15, len(t))
    fr = atrial_organization_features(regular, FS)
    fi = atrial_organization_features(irregular, FS)
    for k in ("periodicity", "spectral_concentration", "phase_locking", "cycle_morph_consistency"):
        assert k in fr and k in fi
    assert fr["periodicity"] > fi["periodicity"], (fr, fi)
    assert fr["spectral_concentration"] > fi["spectral_concentration"], (fr, fi)
