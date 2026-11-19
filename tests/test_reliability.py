"""Stage 3.6.x — ECE test on known miscalibration."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.reliability import expected_calibration_error


def test_perfect_calibration_has_zero_ece():
    # In each bin the predicted confidence equals the empirical accuracy.
    p = np.array([0.6] * 10)
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    assert expected_calibration_error(y, p) < 1e-9


def test_overconfident_model_has_high_ece():
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
    y = np.array([1, 0, 0, 1, 0])
    assert expected_calibration_error(y, p) > 0.2
