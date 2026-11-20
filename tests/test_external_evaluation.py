"""Stage 3.6.9 — external evaluation protocol tests (TDD).

Regression test for the two critical bugs in stage36_external_test.py:
1. training on ALL Chapman rows (including internal test);
2. selecting the threshold from external-set labels.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ecgdet.external_evaluation import (  # noqa: E402
    ExternalTestProtocol,
    assert_probability_direction,
    select_threshold,
    evaluate_predictions,
)


def test_select_threshold_uses_only_validation_labels(monkeypatch):
    seen = {}

    def fake_f1(y_true, y_pred, average):
        seen["y_true"] = list(y_true)
        assert average == "macro"
        return np.ones(len(np.unique(y_true)))

    import ecgdet.external_evaluation as ee
    monkeypatch.setattr(ee, "f1_score", fake_f1)
    thr = select_threshold(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8]))
    assert 0.0 <= thr <= 1.0
    assert seen["y_true"] == [0, 1, 0, 1]


def test_evaluate_predictions_returns_full_metric_set():
    y = np.array([1, 1, 0, 0])
    s = np.array([.9, .8, .3, .1])
    r = evaluate_predictions(y, s, threshold=0.5)
    for k in ("f1_macro", "afib_f1", "afl_f1", "roc_auc", "pr_auc",
              "balanced_accuracy", "brier", "afib_precision", "afib_recall",
              "afl_precision", "afl_recall", "confusion_matrix", "n"):
        assert k in r, k
    assert r["f1_macro"] == pytest.approx(1.0)
    assert r["roc_auc"] == pytest.approx(1.0)


def test_assert_probability_direction_detects_inversion():
    # AFIB label=1 must score higher on average than AFL label=0
    assert_probability_direction(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) is True
    with pytest.raises(ValueError, match="direction"):
        assert_probability_direction(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1]))


def test_protocol_refuses_to_tune_threshold_on_external_labels():
    """Core regression: external y must never reach threshold selection."""
    meta = pd.DataFrame({
        "source": ["chapman"] * 4 + ["ptbxl"] * 2,
        "split": ["train", "train", "val", "val", "test", "test"],
        "label": [0, 1, 0, 1, 1, 0],
    })
    X = np.arange(6).reshape(6, 1).astype(float)

    class DummyModel:
        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return np.column_stack([1 - x[:, 0] / 10, x[:, 0] / 10])

    p = ExternalTestProtocol(meta, X, DummyModel())
    result = p.run(source="chapman", external_sources=["ptbxl"])
    ext = result["external"]["ptbxl"]
    # threshold must come from the val half, never from the external grid
    assert ext["threshold"] == pytest.approx(result["threshold"])
    assert result["train_indices"] == [0, 1]
    assert result["val_indices"] == [2, 3]
    assert ext["indices"] == [4, 5]
    assert "f1_macro" in ext and "roc_auc" in ext
