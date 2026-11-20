"""Stage 3.6.15 — inductive UDA protocol: disjoint adaptation / target test.

The old Stage 3.6.13 script adapted on ALL target records and then evaluated
on the same records.  That is valid transductive UDA but not a test on unseen
target data.  This module enforces a sealed target split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["split_target_domain", "validate_target_isolation"]


def _groups(meta: pd.DataFrame) -> np.ndarray:
    col = "patient_key" if "patient_key" in meta.columns else "record_id"
    return meta[col].astype(str).to_numpy()


def split_target_domain(target_meta: pd.DataFrame, adaptation_fraction: float = 0.5,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split a target domain into unlabeled adaptation and sealed test sets."""
    if not 0 < adaptation_fraction < 1:
        raise ValueError("adaptation_fraction must lie strictly between 0 and 1")
    groups = _groups(target_meta)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_adapt = max(1, min(len(unique) - 1, int(round(len(unique) * adaptation_fraction))))
    adapt_groups = set(unique[:n_adapt])
    adapt = np.where(np.isin(groups, list(adapt_groups)))[0]
    test = np.where(~np.isin(groups, list(adapt_groups)))[0]
    violations = validate_target_isolation(adapt, test, target_meta)
    if violations:
        raise RuntimeError("target split leaks: " + "; ".join(violations))
    return adapt, test


def validate_target_isolation(adaptation_indices, target_test_indices,
                              target_meta: pd.DataFrame) -> list[str]:
    """Return every way the two target subsets overlap."""
    a = np.asarray(list(adaptation_indices), dtype=int)
    t = np.asarray(list(target_test_indices), dtype=int)
    out = []
    if set(a.tolist()) & set(t.tolist()):
        out.append("index overlap between adaptation and target test")
    if len(a) == 0 or len(t) == 0:
        return out
    for col in ("patient_key", "record_id"):
        if col not in target_meta.columns:
            continue
        vals = target_meta[col].astype(str).to_numpy()
        if set(vals[a]) & set(vals[t]):
            out.append(f"{col} overlap between adaptation and target test")
    return out
