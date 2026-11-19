"""Finish merge_all_datasets: build aligned+permuted feature matrices only."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
A = config.PROCESSED_ROOT / "af_afl_merged"
B = config.PROCESSED_ROOT / "shdb_af_windows"
OUT = config.PROCESSED_ROOT / "af_afl_merged_v2"


def load(D, key):
    fm = D / f"features_advanced_{key}.npy"
    nm = D / f"feature_names_{key}.csv"
    if not fm.exists():
        fm = D / f"features_{key}.npy"
    return np.load(fm), pd.read_csv(nm)["name"].tolist()


def main():
    ya = np.load(A / "y.npy"); yb = np.load(B / "y.npy")
    spa = pd.read_csv(A / "split.csv"); spb = pd.read_csv(B / "split.csv")
    order = ["train", "val", "test_chapman", "test_external", "test_shdb"]
    # row order in global X: train(A then B), val(A then B),
    # test_chapman(A), test_external(A), test_shdb(B)
    perm = []
    order = [("train", "A"), ("train", "B"), ("val", "A"), ("val", "B"),
             ("test_chapman", "A"), ("test_external", "A"), ("test_shdb", "B")]
    for split, tag in order:
        sp = spa if tag == "A" else spb
        src_split = "test" if split == "test_shdb" else split
        ids = sp.loc[sp.split == src_split, "idx"].to_numpy()
        if tag == "B":
            ids = ids + len(ya)
        perm.append(ids)
    perm = np.concatenate(perm).astype(int)
    n = len(np.load(OUT / "y.npy"))
    assert len(perm) == n and perm.max() < len(ya) + len(yb)
    yall = np.concatenate([ya, yb])
    np.save(OUT / "y.npy", yall[perm])

    for key, prefix in (("expert", "expert_"), ("v2", "v2_"), ("heat", "heat_")):
        Ma, na = load(A, key); Mb, nb = load(B, key)
        allnames = list(na) + [x for x in nb if x not in na]
        M = np.full((len(Ma) + len(Mb), len(allnames)), np.nan, dtype=np.float32)
        for src, names, off in ((Ma, na, 0), (Mb, nb, len(Ma))):
            for j, name in enumerate(names):
                M[off:off + len(src), allnames.index(name)] = src[:, j]
        np.save(OUT / f"features_{key}.npy", M[perm])
        pd.Series([prefix + x for x in allnames]).to_csv(OUT / f"feature_names_{key}.csv",
                                                         index=False, header=["name"])
        print(key, M[perm].shape, flush=True)
    stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8")) if (OUT / "stats.json").exists() else {}
    stats["features"] = {"expert": len(na), "v2": len(allnames) if key == "v2" else None}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
