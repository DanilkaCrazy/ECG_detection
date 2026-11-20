"""Stage 3.6.9 — lead quality / availability mask tests (TDD)."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ecgdet.lead_quality_audit import (  # noqa: E402
    CANONICAL_LEADS,
    build_lead_mask,
    lead_quality,
    summarize_dataset_leads,
)


def _signal():
    fs = 500
    t = np.arange(5000) / fs
    x = np.tile(np.sin(2 * np.pi * 2 * t), (12, 1))
    x[5] = 0.0            # constant-zero lead
    x[7, :100] = np.nan   # partial NaN
    x[9] += np.random.default_rng(0).normal(0, 5, 5000)  # noisy lead
    return x.astype(np.float32)


def test_lead_quality_flags_zero_nan_and_noisy_leads():
    q = lead_quality(_signal(), 500.0, list(CANONICAL_LEADS))
    assert len(q) == 12
    assert set(["lead", "zero_frac", "nan_frac", "inf_frac", "amplitude", "noise_rms", "usable"]).issubset(q.columns)
    assert not bool(q.loc[q.lead == CANONICAL_LEADS[5], "usable"].iloc[0])
    assert q.loc[q.lead == CANONICAL_LEADS[7], "nan_frac"].iloc[0] > 0
    assert q.loc[q.lead == CANONICAL_LEADS[9], "noise_rms"].iloc[0] > 1.0


def test_build_lead_mask_excludes_bad_leads_but_keeps_good_ones():
    mask = build_lead_mask(_signal(), 500.0, list(CANONICAL_LEADS))
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (12,)
    assert not mask[5], "zero lead must be masked"
    assert mask[0], "clean lead must stay usable"


def test_build_lead_mask_keeps_record_when_only_one_lead_is_good():
    x = np.zeros((12, 5000), dtype=np.float32)
    x[1] = np.sin(np.arange(5000) / 30)
    mask = build_lead_mask(x, 500.0, list(CANONICAL_LEADS))
    assert int(mask.sum()) == 1, "record must not be dropped merely because most leads are missing"


def test_summarize_dataset_leads_aggregates_by_source():
    xs = np.stack([_signal(), _signal()])
    meta = __import__("pandas").DataFrame({"source": ["chapman", "ptbxl"]})
    s = summarize_dataset_leads(xs, meta, 500.0, list(CANONICAL_LEADS))
    assert set(s["source"]) == {"chapman", "ptbxl"}
    assert "mean_usable_leads" in s.columns
