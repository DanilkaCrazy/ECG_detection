"""Stage 3.6.21 — independent-set contract tests (TDD)."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.independent_evaluation import (  # noqa: E402
    independent_set_violations, freeze_artifacts, MIN_PER_CLASS,
)


def _valid(n=MIN_PER_CLASS):
    return pd.DataFrame({
        "record_id": [f"r{i}" for i in range(2 * n)],
        "source": ["extA"] * n + ["extB"] * n,
        "label": [1] * n + [0] * n,
        "patient_id": [f"p{i}" for i in range(2 * n)],
        "expert_label": [1] * n + [0] * n,
        "signal_hash": [f"h{i}" for i in range(2 * n)],
    })


def test_valid_independent_set_passes():
    assert independent_set_violations(_valid()) == []


def test_missing_expert_labels_and_patient_ids_fail_closed():
    d = _valid()
    d.loc[0, "expert_label"] = np.nan
    d.loc[1, "patient_id"] = np.nan
    v = independent_set_violations(d)
    assert any("expert" in x for x in v)
    assert any("patient_id" in x for x in v)


def test_overlap_with_development_hashes_fails_closed():
    d = _valid()
    v = independent_set_violations(d, development_hashes={"h0", "h1"})
    assert any("overlap" in x for x in v)


def test_too_few_records_per_class_fails_closed():
    d = _valid().head(20)
    v = independent_set_violations(d)
    assert any("class" in x for x in v)


def test_single_source_duplicate_hash_and_label_disagreement_fail_closed():
    d = _valid()
    d.loc[0, "label"] = 0          # disagree with expert_label=1
    d.loc[1, "signal_hash"] = d.loc[0, "signal_hash"]  # duplicate
    v = independent_set_violations(d)
    assert any("disagree" in x for x in v)
    assert any("duplicate" in x for x in v)


def test_only_one_source_fails_closed():
    d = _valid()
    d["source"] = "only_one"
    v = independent_set_violations(d)
    assert any("source" in x for x in v)


def test_adjudication_column_resolves_disagreement():
    d = _valid()
    d.loc[0, "label"] = 0
    d["adjudicated_label"] = d["label"]
    v = independent_set_violations(d)
    assert not any("disagree" in x for x in v)


def test_freeze_artifacts_records_threshold_and_hashes(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"abc")
    r = freeze_artifacts(p, ["f1", "f2"], 0.42, seed=7)
    assert r["threshold"] == 0.42
    assert r["model_sha256"] is not None
    assert r["n_features"] == 2
