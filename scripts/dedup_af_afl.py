"""Remove duplicate ECG signals and rebuild leak-free group-aware splits."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ecgdet import config

SRC = config.PROCESSED_ROOT / os.environ.get("AF_AFL_SRC", "af_afl_full10s")
OUT = config.PROCESSED_ROOT / os.environ.get("AF_AFL_OUT", "af_afl_full10s_dedup")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(SRC / "X.npy", mmap_mode="r")
    y = np.load(SRC / "y.npy")
    meta = pd.read_csv(SRC / "meta.csv")
    n = len(y)
    print("input:", n, "records", flush=True)

    seen: dict[str, int] = {}
    keep = []
    dup_groups: list[list[int]] = []
    for i in range(n):
        h = hashlib.md5(np.asarray(X[i], dtype=np.float32).tobytes()).hexdigest()
        if h in seen:
            j = seen[h]
            if not dup_groups or dup_groups[-1][0] != j:
                dup_groups.append([j])
            dup_groups[-1].append(i)
        else:
            seen[h] = i
            keep.append(i)
    keep = np.array(keep, dtype=int)
    print(f"duplicates removed: {n - len(keep)}, groups: {len(dup_groups)}", flush=True)

    yk = y[keep]
    mk = meta.iloc[keep].reset_index(drop=True)
    # group: patient_id when available, else record_id; duplicates of same signal share a group hash
    grp = mk["patient_id"].fillna(mk["record_id"]).astype(str).values
    sig_hash = {}
    for i in keep:
        sig_hash[i] = hashlib.md5(np.asarray(X[i], dtype=np.float32).tobytes()).hexdigest()
    # use record_id as group (Chapman: 1 record = 1 patient); duplicates already removed
    grp = mk["record_id"].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    trl, tmpl = next(gss.split(np.zeros((len(keep), 1)), yk, groups=grp))
    val, te = train_test_split(tmpl, test_size=0.5, stratify=yk[tmpl], random_state=42)
    ids = {"train": keep[trl], "val": keep[val], "test": keep[te]}

    # global X for feature extraction
    gx = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32,
                                   shape=(len(keep), X.shape[1], X.shape[2]))
    for k, gi in enumerate(keep):
        gx[k] = X[gi]
    gx.flush()

    for name, idxs in ids.items():
        out = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+",
                                        dtype=np.float32, shape=(len(idxs), X.shape[1], X.shape[2]))
        for k, gi in enumerate(idxs):
            out[k] = X[gi]
        out.flush()
        np.save(OUT / f"y_{name}.npy", y[idxs])
    np.save(OUT / "y.npy", y[keep])
    mk.to_csv(OUT / "meta.csv", index=False)
    pd.DataFrame({
        "split": sum(([name] * len(ids[name]) for name in ids), []),
        # local positions in the deduplicated X.npy (0..n_kept-1)
        "idx": np.concatenate([trl, val, te]),
    }).to_csv(OUT / "split.csv", index=False)
    stats = {
        "source": str(SRC), "n_input": int(n), "n_kept": int(len(keep)),
        "duplicates_removed": int(n - len(keep)),
        "n_af": int((yk == 1).sum()), "n_afl": int((yk == 0).sum()),
        "n_train": len(ids["train"]), "n_val": len(ids["val"]), "n_test": len(ids["test"]),
        "geometry": {"fs": 500, "n_samp": X.shape[2], "n_leads": 12},
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
