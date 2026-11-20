"""Stage 3.6.9 (P0) — audit existing splits and create a sealed holdout."""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.split_integrity import (  # noqa: E402
    audit_splits, assert_no_leakage, create_locked_holdout,
)

D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "reports" / "metrics" / "stage369"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    audit = audit_splits(sp)
    idx = {f"{name}_idx": np.where(sp.split == name)[0] for name in ("train", "val", "test")}
    overlap = assert_no_leakage(sp, **idx)
    audit["leakage_violations"] = overlap
    audit["split_table"] = "data/processed/af_afl_stage34/split.csv"
    (OUT / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sealed holdout: 4-way, patient-grouped where a real id exists.
    locked = create_locked_holdout(meta, seed=42, holdout_frac=0.20,
                                   val_frac=0.15, test_frac=0.15)
    locked = locked.reset_index(drop=True)
    locked.insert(0, "idx", np.arange(len(locked)))
    lp = D / "split_locked.csv"
    locked[["idx", "split", "source", "label", "patient_key"]].to_csv(lp, index=False)
    la = audit_splits(locked[["idx", "split", "source", "label", "patient_key"]])
    la["fingerprint_locked"] = audit["fingerprint"]
    (OUT / "split_locked_audit.json").write_text(
        json.dumps(la, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("Locked split counts:", la["counts"])
    print("Wrote", lp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
