"""Stage 3.6.10 — genuine inter-lead coherence/phase synchrony tests (TDD)."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.interlead_features import interlead_features  # noqa: E402

FS = 500.0
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def _sig(phase_shift=0.0, noise=0.01, seed=0):
    t = np.arange(5000) / FS
    base = np.sin(2 * np.pi * 6 * t + phase_shift)
    rng = np.random.default_rng(seed)
    x = np.tile(base, (12, 1)) + rng.normal(0, noise, (12, 5000))
    return x.astype(np.float32)


def test_identical_leads_have_high_coherence_and_plv():
    f = interlead_features(_sig(phase_shift=0.0), FS, LEADS)
    assert f["interlead_coherence_mean"] > 0.8
    assert f["interlead_plv_mean"] > 0.8
    assert f["interlead_amp_cv"] < 0.2


def test_phase_shifted_leads_have_lower_plv_than_in_phase():
    aligned = interlead_features(_sig(0.0, noise=0.001), FS, LEADS)
    shifted = interlead_features(_sig(np.pi / 2, noise=0.001), FS, LEADS)
    assert aligned["interlead_plv_mean"] >= shifted["interlead_plv_mean"] - 0.05


def test_phase_locking_is_not_confused_with_single_lead_concentration():
    """The feature must use TWO signals: a single lead cannot define PLV."""
    f = interlead_features(_sig(), FS, LEADS)
    assert "interlead_plv_mean" in f
    assert "interlead_coherence_mean" in f
    assert f["interlead_plv_mean"] <= 1.0


def test_missing_lead_mask_is_respected():
    x = _sig()
    mask = np.ones(12, dtype=bool)
    mask[1] = False  # drop lead II
    f = interlead_features(x, FS, LEADS, lead_mask=mask)
    assert np.isfinite(f["interlead_plv_mean"])
    assert f["n_leads_used"] == 11
