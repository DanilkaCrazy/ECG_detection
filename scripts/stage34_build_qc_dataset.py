"""Stage 3.4 — build FINAL dataset from PASS-only QC manifest.

Fixes P0: review/reject records are REALLY excluded from X/y/features/meta/splits.
Produces af_afl_stage34/ with:
- X.npy, y.npy, meta.csv, split.csv (subset of af_afl_stage31 PASS rows)
- qc_final_manifest.csv (record_id, source, label, status, reasons)
- stats with source+actual manifest hashes and excluded counts per class/source.
Feature arrays are SUBSETTED (not recomputed) by exact row order, with hash checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
SRC = config.PROCESSED_ROOT / "af_afl_stage31"
MET = ROOT / "reports" / "metrics" / "af_afl_stage34"
OUT = config.PROCESSED_ROOT / "af_afl_stage34"
MET.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def h(p):
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            x.update(c)
    return x.hexdigest()


def main() -> int:
    qc = pd.read_csv(ROOT / "reports" / "metrics" / "af_afl_stage33" / "full_signal_qc.csv")
    meta = pd.read_csv(SRC / "meta.csv")
    assert len(qc) == len(meta), (len(qc), len(meta))
    assert (qc["record_id"].astype(str).values == meta["record_id"].astype(str).values).all(), \
        "QC and dataset row order mismatch"
    keep = (qc["status"] == "PASS").to_numpy()
    excluded = (qc[~keep][["record_id", "source", "status", "reasons"]]
                .merge(meta[["record_id", "label", "split"]], on="record_id", how="left"))
    excluded.to_csv(MET / "excluded_records.csv", index=False)
    final = qc[keep][["record_id", "source", "status", "reasons"]].copy()
    final["label"] = meta.loc[keep, "label"].to_numpy()
    final["split"] = meta.loc[keep, "split"].to_numpy()
    final["patient_key"] = meta.loc[keep, "patient_key"].to_numpy()
    final.to_csv(OUT / "qc_final_manifest.csv", index=False)

    X = np.load(SRC / "X.npy", mmap_mode="r")
    y = np.load(SRC / "y.npy")
    mm = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32,
                                   shape=(int(keep.sum()), X.shape[1], X.shape[2]))
    for k, i in enumerate(np.where(keep)[0]):
        mm[k] = X[i]
    mm.flush()
    np.save(OUT / "y.npy", y[keep])
    meta[keep].reset_index(drop=True).to_csv(OUT / "meta.csv", index=False)
    sp = pd.read_csv(SRC / "split.csv")
    sp2 = sp[sp["source"].isin(meta.loc[keep, "source"].unique())].copy()
    # rebuild split by matching kept record_ids (row-level order)
    new_sp = pd.DataFrame({"idx": np.arange(int(keep.sum())),
                           "split": meta.loc[keep, "split"].to_numpy(),
                           "source": meta.loc[keep, "source"].to_numpy(),
                           "label": y[keep],
                           "patient_key": meta.loc[keep, "patient_key"].to_numpy()})
    new_sp.to_csv(OUT / "split.csv", index=False)

    feat_files = {"expert": "features_advanced_expert.npy", "v2": "features_advanced_v2.npy",
                  "heat": "features_heat.npy"}
    hashes = {}
    for name, fn in feat_files.items():
        F = np.load(SRC / fn, mmap_mode="r")
        assert len(F) == len(meta), f"feature rows mismatch: {name}"
        np.save(OUT / fn, np.asarray(F[keep]))
        hashes[name] = h(OUT / fn)
    stats = {
        "n_input": int(len(meta)), "n_kept": int(keep.sum()),
        "n_excluded": int((~keep).sum()),
        "excluded_by_source": excluded.groupby("source").size().to_dict(),
        "excluded_by_label": excluded.groupby("label").size().to_dict(),
        "excluded_by_status": excluded.groupby("status").size().to_dict(),
        "by_split": new_sp.groupby("split").size().to_dict(),
        "by_source": new_sp.groupby("source").size().to_dict(),
        "qc_manifest_sha256": h(OUT / "qc_final_manifest.csv"),
        "source_manifest_sha256": json.loads((SRC / "stats.json").read_text()).get(
            "manifest_final_sha256_actual",
            json.loads((SRC / "stats.json").read_text()).get("manifest_final_sha256", "")),
        "feature_sha256": hashes,
        "row_contract": "X/y/meta/split/features all subset by identical PASS mask in meta row order",
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage 3.4 — QC-filtered dataset", "",
             f"Записей до QC: {stats['n_input']}, PASS: {stats['n_kept']}, "
             f"исключено: {stats['n_excluded']}.", "",
             f"По источникам: {stats['excluded_by_source']}", "",
             f"По классам (1=AFIB): {stats['excluded_by_label']}", "",
             f"По статусам: {stats['excluded_by_status']}", "",
             "REVIEW/REJECT реально удалены из X/y/meta/split и ВСЕХ feature-массивов.", ""]
    (ROOT / "reports" / "AF_AFL_STAGE34_DATA_INTEGRITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
