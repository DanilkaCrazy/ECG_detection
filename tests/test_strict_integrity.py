"""Stage 3.6.1 — strict integrity gate must reject silent zero padding."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from stage36_strict_integrity import check_record


def test_good_record_passes():
    x = np.random.default_rng(0).normal(0, 0.1, (12, 5000))
    assert check_record(x)["gate"] is True


def test_zero_padded_record_is_rejected():
    x = np.zeros((12, 5000))
    x[:, :12] = 1.0
    r = check_record(x)
    assert r["gate"] is False
    assert "channel_nonzero" in r["reasons"]


def test_nan_and_inf_are_rejected():
    x = np.random.default_rng(1).normal(0, 0.1, (12, 5000))
    x[0, 100] = np.nan
    assert "nan" in check_record(x)["reasons"]
    y = np.random.default_rng(2).normal(0, 0.1, (12, 5000))
    y[1, 100] = np.inf
    assert "inf" in check_record(y)["reasons"]


def test_wrong_shape_is_rejected():
    assert check_record(np.zeros((5000, 12)))["gate"] is False
