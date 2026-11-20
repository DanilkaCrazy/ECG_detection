"""Stage 3.6.15 — measure how recognizable the DATASET is from the ECG alone.

This is a diagnostic, not an arrhythmia classifier.  Only signal-derived
features are accepted; record ids, paths, filenames, split ids and diagnostic
labels are never inputs.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .source_signal_audit import audit_signal

__all__ = ["build_source_features", "evaluate_source_classifier"]


def build_source_features(X: np.ndarray, fs: float = 500.0,
                          lead_names: list[str] | None = None) -> np.ndarray:
    """Aggregate signal-only features for source discrimination."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must be (records, leads, samples)")
    names = lead_names or [f"L{i}" for i in range(X.shape[1])]
    cols = ["mean", "std", "rms", "p95_abs", "p99_abs", "nan_fraction",
            "inf_fraction", "zero_fraction", "clipping_fraction",
            "baseline_drift", "hf_ratio"]
    rows = []
    for i in range(len(X)):
        r = audit_signal(X[i], fs, names)
        rows.append([r[c] for c in cols])
    return np.asarray(rows, dtype=np.float32)


def evaluate_source_classifier(features: np.ndarray, sources: np.ndarray,
                               seed: int = 42) -> dict:
    """Train/evaluate a source classifier on a held-out split."""
    from sklearn.model_selection import train_test_split
    f = np.nan_to_num(np.asarray(features, dtype=np.float32))
    s = np.asarray(sources)
    labels = np.unique(s)
    idx = np.arange(len(s))
    tr, te = train_test_split(idx, test_size=0.3, stratify=s, random_state=seed)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed).fit(f[tr], s[tr])
    pred = model.predict(f[te])
    proba = model.predict_proba(f[te])
    return {
        "n_train": len(tr), "n_test": len(te),
        "accuracy": float(accuracy_score(s[te], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(s[te], pred)),
        "macro_f1": float(f1_score(s[te], pred, average="macro")),
        "classes": labels.tolist(),
        "confusion_matrix": confusion_matrix(s[te], pred, labels=labels).tolist(),
        "roc_auc_ovr": float(roc_auc_score((s[te, None] == labels).astype(int), proba,
                                           average="macro", multi_class="ovr"))
        if len(labels) > 2 else float(roc_auc_score((s[te] == labels[0]).astype(int), proba[:, 0])),
    }
