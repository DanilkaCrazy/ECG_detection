"""Stage 3.6.10 — rhythm feature group tests (TDD)."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.rhythm_features import rhythm_features  # noqa: E402


def test_regular_rr_has_low_variability():
    rr = np.full(100, 0.8)
    f = rhythm_features(rr)
    assert f["rr_cv"] == pytest.approx(0.0, abs=1e-6)
    assert f["rr_rmssd"] == pytest.approx(0.0, abs=1e-6)
    assert f["rr_periodicity"] > 0.9


def test_irregular_rr_has_high_variability():
    rng = np.random.default_rng(0)
    rr = rng.uniform(0.4, 1.2, 100)
    f = rhythm_features(rr)
    assert f["rr_cv"] > 0.15
    assert f["rr_rmssd"] > 0.1
    assert f["rr_entropy"] > 0.0


def test_poincare_sd1_sd2_ordering():
    rng = np.random.default_rng(1)
    rr = np.clip(rng.normal(0.8, 0.05, 200), 0.3, 1.5)
    f = rhythm_features(rr)
    # For independent RR samples sd1 and sd2 are approximately equal.
    assert 0 < f["rr_poincare_sd1"] <= 1.2 * f["rr_poincare_sd2"]


def test_short_rr_series_is_handled_without_nan():
    f = rhythm_features(np.array([0.8, 0.9, 0.75]))
    assert all(np.isfinite(v) for v in f.values()), f
    assert set(f) >= {"rr_cv", "rr_rmssd", "rr_entropy", "rr_sample_entropy",
                      "rr_poincare_sd1", "rr_poincare_sd2", "rr_periodicity",
                      "rr_irregularity_index"}
