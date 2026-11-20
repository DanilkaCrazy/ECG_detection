"""Stage 3.6.14 commit 4 — rebuild a provenance-clean dataset.

Only records whose ORIGINAL diagnosis metadata supports an unambiguous AFIB or
AFL label are kept.  Stage 3.6.4 data is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.split_integrity import assert_no_leakage  # noqa: E402

SRC = ROOT / "data" / "processed" / "af_afl_stage34"
D = ROOT / "data" / "processed" / "af_afl_stage3614"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    D.mkdir(parents=True, exist_ok=True)
    prov = pd.read_csv(D / "diagnosis_provenance.csv")
    valid = prov[prov.label_status.isin(["AFIB", "AFL"])].copy().reset_index(drop=True)
    rows = valid.row_in_stage34.to_numpy(dtype=int)
    old_meta = pd.read_csv(SRC / "meta.csv")
    old_split = pd.read_csv(SRC / "split.csv")

    X = np.load(SRC / "X.npy", mmap_mode="r")
    mm = np.lib.format.open_memmap(D / "X.npy", mode="w+", dtype=np.float32,
                                   shape=(len(rows), X.shape[1], X.shape[2]))
    for k, i in enumerate(rows):
        mm[k] = X[i]
    mm.flush()
    y = valid.label.to_numpy(dtype=np.int64)
    np.save(D / "y.npy", y)

    meta = old_meta.iloc[rows].reset_index(drop=True).copy()
    meta["canonical_rhythm"] = valid.canonical_rhythm.to_numpy()
    meta["label_status"] = valid.label_status.to_numpy()
    meta["mapping_version"] = valid.mapping_version.to_numpy()
    meta["raw_diagnosis_codes"] = valid.raw_diagnosis_codes.to_numpy()
    meta["primary_diagnosis"] = valid.primary_diagnosis.to_numpy()
    meta["secondary_diagnoses"] = valid.secondary_diagnoses.to_numpy()
    meta["label_legacy"] = meta.label.to_numpy()
    meta.to_csv(D / "meta.csv", index=False)

    split = old_split.iloc[rows][["split", "source", "label", "patient_key"]].reset_index(drop=True)
    split.insert(0, "idx", np.arange(len(split)))
    sp = split.rename(columns={"split": "split_legacy"})
    # Deterministic 4-way grouped re-split; the sealed holdout is reserved.
    from ecgdet.split_integrity import create_locked_holdout
    meta_for_split = split.copy()
    locked = create_locked_holdout(meta_for_split, seed=42, holdout_frac=0.2,
                                   val_frac=0.15, test_frac=0.15).reset_index(drop=True)
    sp["split"] = locked.split.to_numpy()
    idx = {f"{k}_idx": np.where(sp.split == k)[0] for k in ("train", "val", "test", "holdout")}
    violations = assert_no_leakage(sp, **idx)
    if violations:
        raise RuntimeError("leakage in rebuilt split: " + "; ".join(violations))
    sp.to_csv(D / "split.csv", index=False)

    valid.to_csv(D / "diagnosis_provenance.csv", index=False)
    pd.read_csv(ROOT / "reports" / "metrics" / "stage3614" / "label_audit_full.csv").to_csv(
        D / "label_audit.csv", index=False)
    pd.read_csv(D / "ambiguous_records.csv").to_csv(D / "excluded_records.csv", index=False)

    files = ["X.npy", "y.npy", "meta.csv", "split.csv", "diagnosis_provenance.csv",
             "label_audit.csv", "excluded_records.csv"]
    manifest = {
        "mapping_version": "1.0",
        "source_dataset": "data/processed/af_afl_stage34 (read-only)",
        "n_records": int(len(rows)),
        "by_source": meta.source.value_counts().to_dict(),
        "by_label": meta.label.value_counts().to_dict(),
        "by_split": sp.split.value_counts().to_dict(),
        "sha256": {f: sha256(D / f) for f in files},
        "split_violations": violations,
    }
    (D / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (D / "stats.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "sha256"},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
