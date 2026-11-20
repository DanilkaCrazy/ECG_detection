"""Stage 3.6.9 — leak-free external evaluation protocol (P0).

Replaces ``scripts/stage36_external_test.py``, which had two disqualifying
bugs: it trained on *all* Chapman rows (including the internal test split) and
it picked the classification threshold from the external set's own labels.

Correct protocol implemented here:

    source train  -> fit model
    source val    -> select threshold (and calibrate, if requested)
    [freeze model + threshold]
    source test   -> final in-domain control, one evaluation
    external sets -> inference and metric computation ONLY

External labels are used exclusively in ``evaluate_predictions`` and never in
threshold/model selection.  ``assert_probability_direction`` guards against a
silently inverted class encoding being "repaired" by flipping probabilities.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "select_threshold",
    "evaluate_predictions",
    "assert_probability_direction",
    "ExternalTestProtocol",
]

THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


def select_threshold(
    y_val: np.ndarray,
    score_val: np.ndarray,
    grid: np.ndarray = THRESHOLD_GRID,
) -> float:
    """Pick the threshold on validation labels only (macro-F1 objective)."""
    y = np.asarray(y_val).astype(int)
    s = np.asarray(score_val).astype(float)
    if len(y) == 0:
        raise ValueError("select_threshold got an empty validation set")
    if y.std() == 0:
        raise ValueError("select_threshold needs both classes in validation")
    f1s = np.array([f1_score(y, (s >= t).astype(int), average="macro") for t in grid])
    return float(grid[int(np.argmax(f1s))])


def assert_probability_direction(y_true: np.ndarray, score: np.ndarray) -> bool:
    """AFIB (label 1) must score higher on average than AFL (label 0)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        raise ValueError("assert_probability_direction needs both classes")
    mean_afib = float(s[y == 1].mean())
    mean_afl = float(s[y == 0].mean())
    if mean_afib <= mean_afl:
        raise ValueError(
            "label direction conflict: mean p(AFIB) for AFIB records "
            f"({mean_afib:.3f}) <= mean for AFL records ({mean_afl:.3f}); "
            "fix the data/labels, do not invert probabilities at evaluation"
        )
    return True


def evaluate_predictions(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute the full, fixed metric set for a frozen model and threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    pred = (s >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "afib_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "afl_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "afib_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "afib_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "afl_precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "afl_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(
            (recall_score(y, pred, pos_label=1, zero_division=0)
             + recall_score(y, pred, pos_label=0, zero_division=0)) / 2
        ),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "brier": float(brier_score_loss(y, s)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "prevalence_afib": float(y.mean()),
    }


class ExternalTestProtocol:
    """Train/calibrate on one source, evaluate on locked external sources."""

    def __init__(self, meta: pd.DataFrame, X: np.ndarray, model: Any):
        for col in ("source", "split", "label"):
            if col not in meta.columns:
                raise ValueError(f"meta is missing {col!r}")
        if len(meta) != len(X):
            raise ValueError("meta/X row count mismatch")
        self.meta = meta.reset_index(drop=True)
        self.X = np.asarray(X)
        self.model = model

    def _idx(self, mask: np.ndarray) -> list[int]:
        return [int(v) for v in np.where(mask)[0]]

    def run(
        self,
        source: str,
        external_sources: Iterable[str],
        source_test: bool = True,
    ) -> dict[str, Any]:
        m = self.meta
        src = m.source.astype(str).to_numpy()
        split = m.split.astype(str).to_numpy()

        train_idx = self._idx((src == source) & np.isin(split, ["train"]))
        val_idx = self._idx((src == source) & (split == "val"))
        source_test_idx = self._idx((src == source) & (split == "test"))
        external_idx: dict[str, list[int]] = {
            str(ext): self._idx(src == ext) for ext in external_sources
        }

        if not train_idx or not val_idx:
            raise ValueError(f"source {source!r} needs both train and val rows")
        all_fit = set(train_idx) | set(val_idx)
        for ext, idx in external_idx.items():
            if set(idx) & all_fit:
                raise ValueError(f"external source {ext} leaked into model fitting")

        y = m.label.to_numpy().astype(int)
        self.model.fit(self.X[train_idx], y[train_idx])

        val_score = self.model.predict_proba(self.X[val_idx])[:, 1]
        assert_probability_direction(y[val_idx], val_score)
        threshold = select_threshold(y[val_idx], val_score)

        control: dict[str, Any] | None = None
        if source_test and source_test_idx:
            s = self.model.predict_proba(self.X[source_test_idx])[:, 1]
            control = {"indices": source_test_idx, **evaluate_predictions(y[source_test_idx], s, threshold)}

        external: dict[str, Any] = {}
        for ext, idx in external_idx.items():
            if not idx:
                continue
            s = self.model.predict_proba(self.X[idx])[:, 1]
            external[ext] = {"indices": idx, **evaluate_predictions(y[idx], s, threshold)}

        return {
            "source": source,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "source_test_indices": source_test_idx,
            "threshold": threshold,
            "control": control,
            "external": external,
            "external_labels_used_for_selection": False,
        }
