"""Stage 3.6.9 — split integrity and locked holdout construction (P0).

Guarantees enforced here:

* train / validation / test / holdout never share row indices;
* when a real patient identifier exists, the same patient never crosses a
  split boundary;
* Chapman has no real patient id — ``patient_key`` degenerates to a record id.
  This limitation is reported explicitly and never presented as
  patient-independent validation.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "assert_no_leakage",
    "audit_splits",
    "create_locked_holdout",
    "split_fingerprint",
]

SPLITS = ("train", "val", "test", "holdout")


def _as_indices(values: Iterable[int] | np.ndarray | pd.Series | None) -> list[int]:
    if values is None:
        return []
    return [int(v) for v in np.asarray(values).ravel()]


def split_fingerprint(meta: pd.DataFrame) -> str:
    """Stable SHA256 of the split definition (order-sensitive)."""
    cols = [c for c in ("idx", "patient_key", "source", "label", "split") if c in meta.columns]
    payload = meta[cols].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_no_leakage(
    meta: pd.DataFrame,
    train_idx: Iterable[int] | None = None,
    val_idx: Iterable[int] | None = None,
    test_idx: Iterable[int] | None = None,
    holdout_idx: Iterable[int] | None = None,
) -> list[str]:
    """Return a list of leakage violations (empty list means clean)."""
    sets = {
        "train": set(_as_indices(train_idx)),
        "val": set(_as_indices(val_idx)),
        "test": set(_as_indices(test_idx)),
        "holdout": set(_as_indices(holdout_idx)),
    }
    violations: list[str] = []
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = sets[a] & sets[b]
            if overlap:
                violations.append(
                    f"index overlap {a}/{b}: {len(overlap)} rows (e.g. {sorted(overlap)[:5]})"
                )

    if "patient_key" in meta.columns:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if not sets[a] or not sets[b]:
                    continue
                pa = set(meta.loc[list(sets[a]), "patient_key"].astype(str))
                pb = set(meta.loc[list(sets[b]), "patient_key"].astype(str))
                shared = pa & pb
                if shared:
                    violations.append(
                        f"patient overlap {a}/{b}: {len(shared)} keys (e.g. {sorted(shared)[:5]})"
                    )
    return violations


def audit_splits(meta: pd.DataFrame) -> dict[str, Any]:
    """Audit an existing split table without modifying it."""
    sp = meta if "split" in meta.columns else pd.DataFrame()
    counts: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for split, sub in sp.groupby("split", sort=True):
        counts[str(split)] = int(len(sub))
        if "source" in sub.columns:
            for src, n in sub.groupby("source").size().items():
                by_source.setdefault(str(src), {})[str(split)] = int(n)

    patient_overlap: dict[str, int] = {}
    if "patient_key" in meta.columns:
        groups = {str(k): set(g.patient_key.astype(str)) for k, g in sp.groupby("split")}
        names = sorted(groups)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = groups[names[i]] & groups[names[j]]
                if shared:
                    patient_overlap[f"{names[i]}/{names[j]}"] = len(shared)

    return {
        "n": int(len(meta)),
        "counts": counts,
        "by_source": by_source,
        "patient_overlap": patient_overlap,
        "fingerprint": split_fingerprint(meta),
        "patient_independent": False if patient_overlap else bool(
            "patient_key" in meta.columns and meta.patient_key.nunique() == len(meta)
        ),
    }


def create_locked_holdout(
    meta: pd.DataFrame,
    seed: int = 42,
    holdout_frac: float = 0.2,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Build a fresh, deterministic patient-grouped 4-way split.

    ``holdout`` is sealed: it must never be used for feature selection,
    threshold tuning or model comparison — only for the final report.
    """
    if "patient_key" not in meta.columns:
        raise ValueError("create_locked_holdout requires patient_key")
    rng = np.random.default_rng(seed)
    keys = np.asarray(meta["patient_key"].astype(str))
    unique = np.unique(keys)
    rng.shuffle(unique)
    n = len(unique)
    n_ho = max(1, int(round(n * holdout_frac)))
    n_va = max(1, int(round(n * val_frac)))
    n_te = max(1, int(round(n * test_frac)))

    assignment: dict[str, str] = {}
    pos = 0
    for name, count in (("holdout", n_ho), ("test", n_te), ("val", n_va)):
        for k in unique[pos:pos + count]:
            assignment[k] = name
        pos += count
    for k in unique[pos:]:
        assignment[k] = "train"

    out = meta.copy()
    out["split"] = [assignment[k] for k in keys]
    violations = assert_no_leakage(
        out,
        train_idx=np.where(out.split == "train")[0],
        val_idx=np.where(out.split == "val")[0],
        test_idx=np.where(out.split == "test")[0],
        holdout_idx=np.where(out.split == "holdout")[0],
    )
    if violations:
        raise RuntimeError("generated split leaks: " + "; ".join(violations))
    return out
