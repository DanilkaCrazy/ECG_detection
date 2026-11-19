"""Unit tests for dataset_harmonization (Stage 3.3). Run: python -m pytest scripts/test_harmonization.py -q"""

from __future__ import annotations

import numpy as np
import pytest

from src.ecgdet.dataset_harmonization import (
    CANONICAL_LEADS, HarmonizationStatus, canonicalize_leads,
    harmonize_signal, signal_quality, signal_hash,
)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


LEADS12 = CANONICAL_LEADS


def test_canonicalize_by_name_not_position():
    names = ["AVR", "II", "V1", "I", "III", "AVL", "AVF"] + [f"V{i}" for i in range(2, 7)]
    idx, mask = canonicalize_leads(names, n_channels=len(names))
    assert mask.sum() == 12
    assert idx[3] == 0  # canonical aVR (index 3) -> source channel 0 by NAME, not position 3


def test_missing_lead_mask_out_of_range():
    idx, mask = canonicalize_leads(["I", "II"], n_channels=1)
    assert mask[0] and not mask[1]  # second name has no real channel


def test_mv_physical_no_gain_applied():
    x = np.random.default_rng(0).normal(0, 0.1, (12, 5000))
    out, mask, info = harmonize_signal(x, 500.0, LEADS12, units="mV", signal_kind="physical")
    assert out.shape == (12, 5000) and mask.all()
    assert "gain already applied" in info["unit_conversion"]
    assert info["status"] == HarmonizationStatus.OK


def test_uv_to_mv():
    x = np.random.default_rng(0).normal(0, 100, (12, 5000))
    out, _, info = harmonize_signal(x, 500.0, LEADS12, units="uV", signal_kind="physical")
    assert np.isclose(np.std(out), 0.1, atol=0.01)
    assert info["status"] == HarmonizationStatus.OK


def test_digital_gain_baseline_per_channel():
    rng = np.random.default_rng(0)
    dig = rng.normal(100, 10, (12, 5000))
    gain = np.full(12, 100.0)
    base = np.full(12, 100.0)
    out, _, info = harmonize_signal(dig, 500.0, LEADS12, units="digital",
                                    signal_kind="digital", adc_gain=gain, baseline=base)
    assert np.isclose(np.std(out), 0.1, atol=0.01)
    assert info["status"] == HarmonizationStatus.OK


def test_digital_without_gain_invalid():
    out, mask, info = harmonize_signal(np.zeros((12, 5000)), 500.0, LEADS12,
                                       units="digital", signal_kind="digital")
    assert info["status"] == HarmonizationStatus.INVALID
    assert not mask.any()


def test_unknown_units_not_assumed_mv():
    out, mask, info = harmonize_signal(np.zeros((12, 5000)), 500.0, LEADS12,
                                       units="unknown", signal_kind="physical")
    assert info["status"] == HarmonizationStatus.REQUIRE_REVIEW
    assert not mask.any()


def test_mixed_units():
    x = np.ones((12, 5000)) * 100
    out, _, info = harmonize_signal(x, 500.0, LEADS12, units=["uV"] * 12, signal_kind="physical")
    assert np.allclose(out[0], 0.1)


def test_resample_integer_and_noninteger():
    x = np.random.default_rng(0).normal(0, 0.1, (12, 3600))
    out, _, info = harmonize_signal(x, 360.0, LEADS12, units="mV",
                                    signal_kind="physical", target_fs=500.0, target_len=5000)
    assert out.shape == (12, 5000) and info.get("resampled")
    x2 = np.random.default_rng(0).normal(0, 0.1, (12, 2570))
    out2, _, info2 = harmonize_signal(x2, 257.0, LEADS12, units="mV",
                                      signal_kind="physical", target_fs=500.0, target_len=5000)
    assert out2.shape == (12, 5000)


def test_qc_before_cleaning():
    x = np.array([[1.0, np.nan, np.inf], [0.0, 0.0, 0.0]])
    q = signal_quality(x)
    assert q["nan_frac"] > 0 and q["inf_frac"] > 0


def test_shape_and_hash():
    x = np.random.default_rng(0).normal(0, 0.1, (12, 5000)).astype(np.float32)
    assert signal_hash(x) == signal_hash(x.copy())
    assert signal_hash(x) != signal_hash(x + 1e-3)
