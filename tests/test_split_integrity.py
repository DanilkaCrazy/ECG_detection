"""Stage 3.6.9 — split integrity / locked holdout tests (TDD)."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ecgdet.split_integrity import (  # noqa: E402
    audit_splits,
    assert_no_leakage,
    create_locked_holdout,
)


def _frame(keys, splits, sources=None, labels=None):
    n = len(keys)
    return pd.DataFrame({
        "idx": np.arange(n),
        "patient_key": keys,
        "split": splits,
        "source": sources or ["chapman"] * n,
        "label": labels if labels is not None else [1] * n,
    })


def test_assert_no_leakage_detects_shared_index_and_patient():
    meta = _frame(["p1", "p2", "p3"], ["train", "val", "test"])
    assert assert_no_leakage(meta, [0], [1], [2]) == []

    bad = assert_no_leakage(meta, [0], [0], [2])
    assert any("index" in v or "overlap" in v for v in bad)

    reused = _frame(["p1", "p1", "p3"], ["train", "val", "test"])
    bad2 = assert_no_leakage(reused, [0], [1], [2])
    assert any("patient" in v for v in bad2)


def test_audit_splits_reports_counts_and_missing_splits():
    meta = _frame(["p1", "p2", "p3"], ["train", "val", "test"],
                  sources=["chapman", "chapman", "ptbxl"])
    r = audit_splits(meta)
    assert r["n"] == 3
    assert r["counts"] == {"train": 1, "val": 1, "test": 1}
    assert set(r["by_source"]["ptbxl"]) == {"test"}
    assert r["patient_overlap"] == {}


def test_create_locked_holdout_is_patient_disjoint_and_reproducible():
    keys = [f"p{i // 2}" for i in range(20)]  # 10 patients, 2 records each
    meta = _frame(keys, ["train"] * 20)
    a = create_locked_holdout(meta, seed=7, holdout_frac=0.25)
    b = create_locked_holdout(meta, seed=7, holdout_frac=0.25)
    assert a.equals(b), "holdout must be deterministic for a fixed seed"
    assert set(a.split) >= {"train", "val", "test", "holdout"}
    tr = set(a.loc[a.split == "train", "patient_key"])
    ho = set(a.loc[a.split == "holdout", "patient_key"])
    assert tr.isdisjoint(ho)
    assert len(ho) > 0
