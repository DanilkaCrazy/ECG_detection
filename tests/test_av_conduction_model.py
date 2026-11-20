"""Stage 3.6.20 — AV conduction physiological features (TDD)."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.av_conduction_model import av_conduction_features, FEATURE_NAMES  # noqa: E402


def _rr_afl(ratio=2, cycle=0.24, n=60, jitter=0.002):
    """Regular atrial cycle with fixed AV block -> RR = ratio * cycle."""
    rng = np.random.default_rng(0)
    return np.full(n, ratio * cycle) + rng.normal(0, jitter, n)


def _rr_afl_mixed(n=60):
    """Mixed 2:1 and 3:1 conduction (multi-level block)."""
    rng = np.random.default_rng(1)
    rr = np.array([2 * 0.24 if i % 3 else 3 * 0.24 for i in range(n)])
    return rr + rng.normal(0, 0.003, n)


def _rr_afib(n=60):
    rng = np.random.default_rng(2)
    return np.clip(rng.lognormal(np.log(0.7), 0.45, n), 0.25, 2.0)


def test_feature_set_is_complete_and_finite():
    f = av_conduction_features(_rr_afib())
    assert set(f) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())
    assert 0 <= f["av_model_reliability"] <= 1


def test_fixed_av_block_has_low_model_error():
    regular = av_conduction_features(_rr_afl())
    irregular = av_conduction_features(_rr_afib())
    assert regular["av_model_error"] < irregular["av_model_error"]
    assert regular["av_ratio_consistency"] > irregular["av_ratio_consistency"]


def test_multi_level_block_is_detected():
    f = av_conduction_features(_rr_afl_mixed())
    assert f["av_n_levels"] >= 2
    assert f["multi_level_block_prob"] > 0.2


def test_sample_entropy_distinguishes_regular_from_irregular_rr():
    regular = av_conduction_features(_rr_afl(jitter=0.00001))
    irregular = av_conduction_features(_rr_afib())
    assert regular["rr_sample_entropy"] < irregular["rr_sample_entropy"]


def test_atrial_cycle_is_near_the_true_value():
    f = av_conduction_features(_rr_afl(ratio=2, cycle=0.24))
    assert abs(f["atrial_cycle_s"] - 0.24) < 0.02
