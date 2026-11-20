"""Stage 3.6.21 — contract for a genuinely independent final test set.

The current development holdout was reused for model comparison, so it cannot
serve as the final evaluation.  This module FAILS CLOSED until a new set with
expert labels, true patient ids and no overlap with development data exists.
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import pandas as pd

__all__ = ["independent_set_violations", "freeze_artifacts", "MIN_PER_CLASS"]

MIN_PER_CLASS = 100
REQUIRED_COLUMNS = ("record_id", "source", "label", "patient_id",
                    "expert_label", "signal_hash")


def independent_set_violations(meta: pd.DataFrame,
                               development_hashes: set[str] | None = None) -> list[str]:
    """Return all reasons why this set cannot be used as final independent test."""
    out = []
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        return [f"missing required columns: {missing}"]
    if meta.record_id.duplicated().any():
        out.append("duplicate record_id")
    if meta.patient_id.isna().any() or (meta.patient_id.astype(str) == "").any():
        out.append("true patient_id missing for some records")
    if meta.expert_label.isna().any():
        out.append("expert labels missing")
    if not set(meta.expert_label.dropna().unique()) <= {0, 1}:
        out.append("expert labels must be binary 0/1")
    counts = meta.expert_label.value_counts()
    for lab in (0, 1):
        if int(counts.get(lab, 0)) < MIN_PER_CLASS:
            out.append(f"class {lab} has {int(counts.get(lab, 0))} < {MIN_PER_CLASS} records")
    if development_hashes is not None:
        overlap = set(meta.signal_hash.astype(str)) & set(development_hashes)
        if overlap:
            out.append(f"{len(overlap)} signal hashes overlap with development data")
    return out


def freeze_artifacts(model_path: str | pathlib.Path, feature_names: list[str],
                     threshold: float, seed: int = 42) -> dict[str, Any]:
    """Hash the frozen model / features / threshold for a single final run."""
    p = pathlib.Path(model_path)
    h = hashlib.sha256()
    if p.exists():
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    fn_h = hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()
    return {
        "model_path": str(p), "model_sha256": h.hexdigest() if p.exists() else None,
        "feature_names_sha256": fn_h, "n_features": len(feature_names),
        "threshold": float(threshold), "seed": seed,
        "note": "No tuning on the independent set after this freeze.",
    }
