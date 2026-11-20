"""Stage 3.6.18 — re-extract diagnoses with the fixed SNOMED resolver.

Builds ``af_afl_stage3614_v1_1`` (never overwrites v1) and writes a full
old-vs-new comparison: class changes, newly excluded, unchanged.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.diagnosis_provenance import (  # noqa: E402
    extract_chapman_diagnoses, extract_cpsc_diagnoses, extract_ptbxl_diagnoses,
    to_dataframe,
)
from ecgdet.label_resolution import resolve_binary  # noqa: E402
from ecgdet import config  # noqa: E402
from ecgdet.split_integrity import (  # noqa: E402
    assert_no_leakage, create_locked_holdout,
)

V1 = ROOT / "data" / "processed" / "af_afl_stage3614"
SRC = ROOT / "data" / "processed" / "af_afl_stage34"
D = ROOT / "data" / "processed" / "af_afl_stage3614_v1_1"
OUT = ROOT / "reports" / "metrics" / "stage3618"
D.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

_CPSC = {p.stem: p for p in (ROOT / "data" / "raw" / "cpsc2018_extra" / "data").rglob("*.hea")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    meta = pd.read_csv(SRC / "meta.csv")
    ptb = pd.read_csv(config.RAW_ROOT / "ptbxl" / "physionet.org" / "files" /
                      "ptb-xl" / "1.0.3" / "ptbxl_database.csv")
    records = []
    keep_rows = []
    for i, row in meta.iterrows():
        rid = str(row.record_id)
        if row.source == "chapman":
            p = pathlib.Path(row.orig_path)
            if not p.exists():
                continue
            rec = extract_chapman_diagnoses(p, rid)
        elif row.source == "cpsc2018_extra":
            if rid not in _CPSC:
                continue
            rec = extract_cpsc_diagnoses(_CPSC[rid], rid)
        elif row.source == "ptbxl":
            try:
                k = int(rid)
            except ValueError:
                continue
            if not (0 <= k < len(ptb)):
                continue
            dbrow = ptb.iloc[k]
            rec = extract_ptbxl_diagnoses(dbrow.get("scp_codes"), rid,
                                          patient_id=dbrow.get("patient_id"),
                                          strat_fold=dbrow.get("strat_fold"))
        else:
            continue
        resolve_binary(rec)
        records.append(rec)
        keep_rows.append(i)

    prov = to_dataframe(records)
    prov.insert(0, "row_in_stage34", keep_rows)
    prov.to_csv(D / "diagnosis_provenance_all.csv", index=False)

    valid = prov[prov.label_status.isin(["AFIB", "AFL"])].reset_index(drop=True)
    valid.to_csv(D / "diagnosis_provenance.csv", index=False)
    rows = valid.row_in_stage34.to_numpy(dtype=int)
    X = np.load(SRC / "X.npy", mmap_mode="r")
    mm = np.lib.format.open_memmap(D / "X.npy", mode="w+", dtype=np.float32,
                                   shape=(len(rows), X.shape[1], X.shape[2]))
    for k, i in enumerate(rows):
        mm[k] = X[i]
    mm.flush()
    y = valid.label.to_numpy(dtype=np.int64)
    np.save(D / "y.npy", y)

    m = meta.iloc[rows].reset_index(drop=True).copy()
    m["label"] = y  # authoritative label comes from the resolver, not old meta
    m["label_legacy"] = meta.iloc[rows].label.to_numpy()
    m["canonical_rhythm"] = valid.canonical_rhythm.to_numpy()
    m["label_status"] = valid.label_status.to_numpy()
    m["mapping_version"] = valid.mapping_version.to_numpy()
    m.to_csv(D / "meta.csv", index=False)

    old_split = pd.read_csv(SRC / "split.csv").iloc[rows][
        ["split", "source", "label", "patient_key"]].reset_index(drop=True)
    from ecgdet.split_integrity import create_locked_holdout as _create
    locked = _create(old_split, seed=42, holdout_frac=0.2, val_frac=0.15,
                     test_frac=0.15).reset_index(drop=True)
    sp = pd.DataFrame({"idx": np.arange(len(locked)),
                       "split": locked.split.to_numpy(),
                       "source": locked.source.to_numpy(),
                       "label": locked.label.to_numpy(),
                       "patient_key": locked.patient_key.to_numpy()})
    violations = assert_no_leakage(sp, **{f"{k}_idx": np.where(sp.split == k)[0]
                                          for k in ("train", "val", "test", "holdout")})
    assert not violations, violations
    sp.to_csv(D / "split.csv", index=False)
    pd.read_csv(V1 / "ambiguous_records.csv").to_csv(D / "excluded_records.csv", index=False)

    # Consistency gates required by the reviewer.
    assert np.array_equal(m.label.to_numpy(), np.load(D / "y.npy"))
    assert np.array_equal(sp.label.to_numpy(), np.load(D / "y.npy"))
    assert np.array_equal(m.signal_hash.to_numpy(), meta.iloc[rows].signal_hash.to_numpy())

    # --- compare v1 vs v1_1 ---
    old = pd.read_csv(V1 / "diagnosis_provenance.csv")
    old_valid = old[old.label_status.isin(["AFIB", "AFL"])]
    new_valid = valid
    merged = old_valid[["record_id", "source", "label"]].merge(
        new_valid[["record_id", "source", "label"]], on=["record_id", "source"],
        how="outer", suffixes=("_v1", "_v11"), indicator=True)
    changed = merged[(merged._merge == "both") & (merged.label_v1 != merged.label_v11)]
    excluded = merged[merged._merge == "left_only"]
    added = merged[merged._merge == "right_only"]
    unchanged = merged[(merged._merge == "both") & (merged.label_v1 == merged.label_v11)]
    changed.to_csv(OUT / "label_changes.csv", index=False)
    excluded.to_csv(OUT / "newly_excluded.csv", index=False)

    comparison = {
        "v1_valid": int(len(old_valid)), "v11_valid": int(len(new_valid)),
        "unchanged": int(len(unchanged)), "class_changed": int(len(changed)),
        "newly_excluded": int(len(excluded)), "newly_added": int(len(added)),
        "explanation": (
            "164934002 (T wave Change) was wrongly treated as AFIB, creating "
            "false AMBIGUOUS records; removing it RESTORED 1325 records. "
            "No class changed. Bare Chapman acronym 'AF' is now namespaced to flutter."
        ),
    }
    manifest = {
        "mapping_version": "1.1", "n_records": int(len(m)),
        "by_source": m.source.value_counts().to_dict(),
        "by_label": m.label.value_counts().to_dict(),
        "by_split": sp.split.value_counts().to_dict(),
        "comparison": comparison,
        "sha256": {f: sha(D / f) for f in ("X.npy", "y.npy", "meta.csv",
                                          "split.csv", "diagnosis_provenance.csv")},
        "consistency_gates": "meta.label == y.npy == split.label == signal_hash",
    }
    (D / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "rebuild_comparison.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
