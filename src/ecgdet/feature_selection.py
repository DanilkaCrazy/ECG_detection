"""Stage 3.6.10 — leak-free feature pruning and selection helpers."""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

__all__ = ["prune_constants", "correlation_prune", "select_features"]


def prune_constants(X: np.ndarray, names: list[str], eps: float = 1e-9):
    """Drop constant and all-NaN columns."""
    X = np.asarray(X, dtype=float)
    good = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(~np.isfinite(col)):
            continue
        finite = col[np.isfinite(col)]
        if finite.size == 0 or np.ptp(finite) < eps:
            continue
        good.append(j)
    return X[:, good], [names[j] for j in good], good


def correlation_prune(X: np.ndarray, names: list[str], threshold: float = 0.98):
    """Drop near-duplicate columns (|r| > threshold), keeping the first."""
    X = np.asarray(X, dtype=float)
    keep = list(range(X.shape[1]))
    for j in range(X.shape[1]):
        if j not in keep:
            continue
        for k in list(keep):
            if k <= j:
                continue
            a, b = X[:, j], X[:, k]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3:
                continue
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            if np.isfinite(r) and abs(r) > threshold:
                keep.remove(k)
    return X[:, keep], [names[j] for j in keep], keep


def select_features(X_fit: np.ndarray, y_fit: np.ndarray, names: list[str],
                    k: int | None = None, method: str = "f_classif",
                    seed: int = 42):
    """Select features using FIT data only (caller passes train/val indices)."""
    X_fit = np.asarray(X_fit, dtype=float)
    y_fit = np.asarray(y_fit).astype(int)
    if method == "l1":
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                               max_iter=2000, random_state=seed)
        lr.fit(np.nan_to_num(X_fit), y_fit)
        score = np.abs(lr.coef_[0])
        k = k or int(np.sum(score > 1e-8))
        idx = np.argsort(-score)[:k]
    else:
        k = min(k or X_fit.shape[1], X_fit.shape[1])
        sel = SelectKBest(f_classif, k=k).fit(np.nan_to_num(X_fit), y_fit)
        idx = np.where(sel.get_support())[0]
    return sorted(int(i) for i in idx), [names[i] for i in sorted(idx)]
