"""Stage 3.6.15 — normalisation axis and train-fitted statistics tests."""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.signal_normalization import (  # noqa: E402
    normalize_per_lead_time, normalize_interlead, normalize_robust,
    fit_train_stats, apply_train_stats,
)


def test_per_lead_time_normalisation_makes_each_lead_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, (5, 12, 1000)).astype(np.float32)
    z = normalize_per_lead_time(x)
    assert np.allclose(z.mean(axis=-1), 0, atol=1e-4)
    assert np.allclose(z.std(axis=-1), 1, atol=1e-4)


def test_interlead_is_a_different_transform_not_a_synonym():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (5, 12, 1000)).astype(np.float32)
    a = normalize_per_lead_time(x)
    b = normalize_interlead(x)
    assert not np.allclose(a, b)


def test_robust_normalisation_is_bounded():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (2, 12, 500))
    x[0, 0, 0] = 1e6
    z = normalize_robust(x)
    assert np.isfinite(z).all()
    assert np.abs(z).max() <= 20


def test_train_fitted_stats_are_identical_across_splits():
    rng = np.random.default_rng(3)
    train = rng.normal(0, 1, (100, 12, 500))
    a = rng.normal(5, 2, (20, 12, 500))
    b = rng.normal(-5, 0.5, (20, 12, 500))
    stats = fit_train_stats(train)
    before = {k: v.copy() for k, v in stats.items()}
    za, zb = apply_train_stats(a, stats), apply_train_stats(b, stats)
    za2 = apply_train_stats(a, stats)
    # Applying to other splits must never modify the frozen train statistics.
    assert np.allclose(za, za2)
    assert np.allclose(stats["scale"], before["scale"])
    assert np.allclose(stats["median"], before["median"])
    assert za.shape == a.shape and zb.shape == b.shape
