"""Build a STRICT AFIB/AFL dataset (primary diagnosis code only) from the
deduplicated 10-second dataset, reusing precomputed feature matrices.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
SRC = config.PROCESSED_ROOT / "af_afl_full10s_dedup"
OUT = config.PROCESSED_ROOT / "af_afl_strict"
AFIB, AFL = "164889003", "164890007"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SRC / "meta.csv")
    man = pd.read_csv(ROOT / "reports" / "manifests" / "chapman_manifest.csv").set_index("record")
    first = {}
    for rec, v in man["dx_codes"].items():
        cs = [c for c in re.split(r"[,;\s]+", str(v).strip("[]'\"")) if c]
        first[rec] = cs[0] if cs else ""
    meta["first_dx"] = meta["record_id"].map(first)
    strict = meta[meta["first_dx"].isin([AFIB, AFL])].copy().reset_index(drop=True)
    idx = strict.index.to_numpy()
    y = np.where(strict["first_dx"] == AFIB, 1, 0).astype(int)
    print(f"strict: {len(strict)} records, AFIB={int(y.sum())}, AFL={int((y==0).sum())}", flush=True)

    # rebuild group-aware split on strict subset
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    grp = strict["record_id"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr, tmp = next(gss.split(np.zeros((len(strict), 1)), y, groups=grp))
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    Xall = np.load(SRC / "X.npy", mmap_mode="r")
    for name, local in (("train", tr), ("val", va), ("test", te)):
        glob = idx[local]
        dst = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+", dtype=np.float32,
                                        shape=(len(glob), Xall.shape[1], Xall.shape[2]))
        for k, g in enumerate(glob):
            dst[k] = Xall[g]
        dst.flush()
        np.save(OUT / f"y_{name}.npy", y[local])
    # global arrays: feature matrices already computed for SRC; subset by local index
    for fname in ("features_advanced_expert.npy", "features_advanced_v2.npy"):
        M = np.load(SRC / fname, mmap_mode="r")
        sub = np.asarray(M[idx])
        np.save(OUT / fname, sub)
    # combined
    e = np.load(OUT / "features_advanced_expert.npy")
    v = np.load(OUT / "features_advanced_v2.npy")
    np.save(OUT / "features_combined.npy", np.concatenate([e, v], axis=1))
    np.save(OUT / "y.npy", y)
    strict.to_csv(OUT / "meta.csv", index=False)
    pd.DataFrame({
        "split": ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te),
        "idx": np.concatenate([tr, va, te]),
    }).to_csv(OUT / "split.csv", index=False)
    stats = {"n": len(strict), "n_afib": int(y.sum()), "n_afl": int((y == 0).sum()),
             "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "label": "primary #Dx code only"}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
