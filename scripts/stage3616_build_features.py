"""Stage 3.6.16 — build features for the provenance-clean dataset.

Existing feature rows are SUBSET, never recomputed blindly: every row's
``signal_hash`` must match between stage34 and stage3614 before a feature is
accepted.  This satisfies the Stage 3.6.14 acceptance criterion.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "af_afl_stage34"
S36 = ROOT / "data" / "processed" / "af_afl_stage36"
S3610 = ROOT / "data" / "processed" / "af_afl_stage3610"
D = ROOT / "data" / "processed" / "af_afl_stage3616"
D.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    meta = pd.read_csv(D.parent / "af_afl_stage3614" / "meta.csv")
    prov = pd.read_csv(D.parent / "af_afl_stage3614" / "diagnosis_provenance.csv")
    rows = prov.row_in_stage34.to_numpy(dtype=int)
    old = pd.read_csv(SRC / "meta.csv")

    # HARD GATE: hash agreement for every selected row.
    bad = np.where(old.signal_hash.to_numpy()[rows] != meta.signal_hash.to_numpy())[0]
    if len(bad):
        raise RuntimeError(f"signal hash mismatch for {len(bad)} rows; refusing to reuse features")

    expert = np.load(SRC / "features_advanced_expert.npy", mmap_mode="r")
    grouped = np.load(S3610 / "X.npy", mmap_mode="r")
    gnames = pd.read_csv(S3610 / "feature_names.csv")
    X = np.column_stack([np.asarray(expert[rows], dtype=np.float32),
                         np.asarray(grouped[rows], dtype=np.float32)]).astype(np.float32)
    names = [f"X_expert_{i}" for i in range(expert.shape[1])] + gnames.name.astype(str).tolist()
    groups = ["X"] * expert.shape[1] + gnames.group.astype(str).tolist()
    assert X.shape[1] == len(names) == len(groups)

    np.save(D / "X.npy", X)
    pd.DataFrame({"name": names, "group": groups}).to_csv(D / "feature_names.csv", index=False)
    meta.to_csv(D / "meta.csv", index=False)
    pd.read_csv(D.parent / "af_afl_stage3614" / "split.csv").to_csv(D / "split.csv", index=False)

    stats = {
        "n_records": int(len(X)), "n_features": int(X.shape[1]),
        "by_group": {g: int(sum(x == g for x in groups)) for g in sorted(set(groups))},
        "hash_mismatches": int(len(bad)), "nan_frac": float(np.isnan(X).mean()),
        "feature_sha256": sha(D / "X.npy"),
    }
    (D / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
