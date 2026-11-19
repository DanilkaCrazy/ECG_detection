"""Stage 3.4 RED tests — bugs found in repository review.

These tests MUST fail on current harmonize_signal implementation:
1. NaN/Inf in input must NOT leave status OK
2. missing canonical leads must NOT leave status OK
3. too-short signal padded to target length must be flagged
4. single global "uV" unit must convert ALL channels
5. digital signal with per-channel gain/baseline must convert correctly
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ecgdet.dataset_harmonization import (
    CANONICAL_LEADS, HarmonizationStatus, harmonize_signal,
)


def test_nan_inf_not_ok():
    x = np.random.default_rng(0).normal(0, 0.1, (12, 5000))
    x[0, 100] = np.nan
    x[1, 200] = np.inf
    _, _, info = harmonize_signal(x, 500.0, CANONICAL_LEADS, units="mV", signal_kind="physical")
    assert info["status"] != HarmonizationStatus.OK


def test_missing_leads_not_ok():
    x = np.random.default_rng(0).normal(0, 0.1, (3, 5000))
    _, mask, info = harmonize_signal(x, 500.0, ["I", "II", "V1"], units="mV", signal_kind="physical")
    assert info["status"] != HarmonizationStatus.OK
    assert mask.sum() == 3


def test_too_short_flagged():
    x = np.random.default_rng(0).normal(0, 0.1, (12, 1000))
    _, _, info = harmonize_signal(x, 500.0, CANONICAL_LEADS, units="mV", signal_kind="physical")
    assert info["status"] != HarmonizationStatus.OK


def test_single_uv_unit_converts_all_channels():
    x = np.ones((12, 5000)) * 1000.0
    out, _, info = harmonize_signal(x, 500.0, CANONICAL_LEADS, units="uV", signal_kind="physical")
    assert np.allclose(out, 1.0, atol=1e-3), out[:, 0]


def test_per_channel_gain_baseline():
    x = np.ones((12, 5000)) * 200.0
    gain = np.linspace(100, 200, 12)
    base = np.full(12, 100.0)
    out, _, info = harmonize_signal(x, 500.0, CANONICAL_LEADS, units="digital",
                                    signal_kind="digital", adc_gain=gain, baseline=base)
    assert np.allclose(out[0], 1.0, atol=1e-3)
    assert np.allclose(out[-1], 0.5, atol=1e-3)


def test_gain_length_mismatch_invalid():
    x = np.zeros((12, 5000))
    _, _, info = harmonize_signal(x, 500.0, CANONICAL_LEADS, units="digital",
                                  signal_kind="digital", adc_gain=np.ones(3))
    assert info["status"] == HarmonizationStatus.INVALID
