"""Stage 3.6.22 — align v1.1 provenance CSV with its 7505-record dataset.

The rebuild script saved ALL extracted records into diagnosis_provenance.csv,
while X/y/meta/split contain only valid AFIB/AFL rows.  Keep the full table
under a separate name and make diagnosis_provenance.csv authoritative.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage3614_v1_1"
MET = ROOT / "reports" / "metrics" / "stage3622"


def main() -> int:
    full = pd.read_csv(D / "diagnosis_provenance.csv")
    full.to_csv(D / "diagnosis_provenance_all.csv", index=False)
    valid = full[full.label_status.isin(["AFIB", "AFL"])].reset_index(drop=True)
    valid.to_csv(D / "diagnosis_provenance.csv", index=False)
    meta = pd.read_csv(D / "meta.csv")
    y = np.load(D / "y.npy")
    sp = pd.read_csv(D / "split.csv")
    assert len(valid) == len(meta) == len(y) == len(sp), (
        len(valid), len(meta), len(y), len(sp))
    assert np.array_equal(valid.label.to_numpy(), y)
    report = {
        "n_all": int(len(full)), "n_valid": int(len(valid)),
        "n_dataset": int(len(meta)),
        "consistency": "diagnosis_provenance == meta == y == split",
    }
    (MET / "provenance_alignment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
