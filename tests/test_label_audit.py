"""Stage 3.6.9 — label audit tests (TDD, RED first)."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ecgdet.label_audit import (  # noqa: E402
    CLASS_BY_RHYTHM,
    SOURCE_RHYTHM_CODES,
    build_label_table,
    audit_labels,
    check_label_direction,
)


def _meta(rows):
    return pd.DataFrame(rows)


def test_class_by_rhythm_is_consistent_across_all_sources():
    assert CLASS_BY_RHYTHM == {"AF": 1, "AFIB": 1, "AFL": 0}
    for source, rules in SOURCE_RHYTHM_CODES.items():
        assert set(rules["afib"]) | set(rules["afl"]) <= set(CLASS_BY_RHYTHM), source


def test_build_label_table_flags_correct_and_conflicting_records():
    m = _meta([
        {"source": "chapman", "record_id": "C1", "rhythm": "AF", "label": 1},
        {"source": "ptbxl", "record_id": "P1", "rhythm": "AFIB", "label": 1},
        {"source": "cpsc2018_extra", "record_id": "X1", "rhythm": "AFL", "label": 0},
        {"source": "chapman", "record_id": "C2", "rhythm": "AFL", "label": 1},  # conflict
        {"source": "ptbxl", "record_id": "P2", "rhythm": "XXXX", "label": 0},  # unknown
    ])
    t = build_label_table(m)
    assert len(t) == len(m)
    assert set(t["status"]) == {"OK", "CONFLICT", "UNKNOWN_CODE"}
    assert t.loc[t.record_id == "C2", "status"].iloc[0] == "CONFLICT"
    assert t.loc[t.record_id == "P2", "status"].iloc[0] == "UNKNOWN_CODE"
    assert t.loc[t.record_id == "C1", "expected_label"].iloc[0] == 1


def test_audit_labels_detects_conflicts_and_missing_codes():
    m = _meta([
        {"source": "chapman", "record_id": "C1", "rhythm": "AF", "label": 1},
        {"source": "chapman", "record_id": "C2", "rhythm": "AF", "label": 0},
        {"source": "ptbxl", "record_id": "P1", "rhythm": "AFIB", "label": 1},
    ])
    r = audit_labels(m)
    assert r["n_records"] == 3
    assert r["n_conflicts"] == 1
    assert "chapman" in r["by_source"]
    assert r["direction_ok"] is True  # mapping AF->1, AFIB->1, AFL->0 is consistent


def test_check_label_direction_rejects_inverted_source():
    """A source where rhythm=AFL is labelled 1 must be flagged, not silently accepted."""
    m = _meta([
        {"source": "chapman", "record_id": "C1", "rhythm": "AF", "label": 1},
        {"source": "bad_source", "record_id": "B1", "rhythm": "AFL", "label": 1},
    ])
    with pytest.raises(ValueError, match="label direction"):
        check_label_direction(m)
