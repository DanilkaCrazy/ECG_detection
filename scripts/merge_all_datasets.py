"""Merge Chapman+PTB-XL+CPSC dataset with SHDB-AF windows (aligned features)."""

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


def aligned(datasets, key, prefix):
    mats, names_list = [], []
    for D in datasets:
        fm = D / f"features_advanced_{key}.npy"
        nm = D / f"feature_names_{key}.csv"
        if not fm.exists():
            fm = D / f"features_{key}.npy"
        if not nm.exists():
            nm = D / f"feature_names_{key}.csv"
        mats.append(np.load(fm))
        names_list.append(pd.read_csv(nm)["name"].tolist())
    # union of names in first dataset order + extras
    base = names_list[0]
    extra = [n for names in names_list[1:] for n in names if n not in base]
    allnames = base + extra
    pos = [{n: i for i, n in enumerate(names)} for names in names_list]
    out = np.full((sum(len(m) for m in mats), len(allnames)), np.nan, dtype=np.float32)
    off = 0
    for M, p in zip(mats, pos):
        for j, n in enumerate(p):
            out[off:off + len(M), allnames.index(n)] = M[:, j]
        off += len(M)
    return out, [f"{prefix}{n}" for n in allnames]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ya = np.load(A / "y.npy"); yb = np.load(B / "y.npy")
    spa = pd.read_csv(A / "split.csv"); spb = pd.read_csv(B / "split.csv")
    ma = pd.read_csv(A / "meta.csv"); mb = pd.read_csv(B / "meta.csv")
    XA = np.load(A / "X.npy", mmap_mode="r")
    XB = np.load(B / "X.npy", mmap_mode="r")

    order = ["train", "val", "test_chapman", "test_external", "test_shdb"]
    mapping = {
        "train": [("A", spa, "train"), ("B", spb, "train")],
        "val": [("A", spa, "val"), ("B", spb, "val")],
        "test_chapman": [("A", spa, "test_chapman")],
        "test_external": [("A", spa, "test_external")],
        "test_shdb": [("B", spb, "test")],
    }
    all_y, all_meta = [], []
    for name in order:
        chunks = []
        for tag, sp, split in mapping[name]:
            ids = sp.loc[sp.split == split, "idx"].to_numpy()
            y = ya if tag == "A" else yb
            meta = ma if tag == "A" else mb
            X = XA if tag == "A" else XB
            chunks.append((ids, y, meta, X))
        n = sum(len(c[0]) for c in chunks)
        dst = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+", dtype=np.float32,
                                        shape=(n, 12, 5000))
        pos = 0
        rows = []
        for ids, y, meta, X in chunks:
            for k, i in enumerate(ids):
                dst[pos] = X[i]; pos += 1
            sub = meta.iloc[ids].reset_index(drop=True)
            sub["label"] = y[ids]
            rows.append(sub)
        dst.flush()
        mm = pd.concat(rows, ignore_index=True)
        mm.to_csv(OUT / f"meta_{name}.csv", index=False)
        np.save(OUT / f"y_{name}.npy", mm["label"].to_numpy())
        all_y.append(mm["label"].to_numpy()); all_meta.append(mm)
        print(name, n, "AFIB", int((mm["label"] == 1).sum()), flush=True)

    y = np.concatenate(all_y)
    np.save(OUT / "y.npy", y)
    pd.concat(all_meta, ignore_index=True).to_csv(OUT / "meta.csv", index=False)
    # global X for feature extraction
    gx = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32, shape=(len(y), 12, 5000))
    pos = 0
    for name in order:
        a = np.load(OUT / f"X_{name}.npy", mmap_mode="r")
        gx[pos:pos + len(a)] = a; pos += len(a)
    gx.flush()
    split_rows = []
    for name in order:
        split_rows += [name] * len(np.load(OUT / f"y_{name}.npy"))
    pd.DataFrame({"split": split_rows, "idx": np.arange(len(y))}).to_csv(OUT / "split.csv", index=False)

    # aligned feature matrices in the SAME row order as global X
    feats = {}
    e, en = aligned([A, B], "expert", "expert_")
    v, vn = aligned([A, B], "v2", "v2_")
    h, hn = aligned([A, B], "heat", "heat_")
    # reorder each feature block to global order (A order then B order per split group)
    # current aligned order is A_all then B_all; build permutation
    perm = []
    for tag, D, sp in (("A", A, spa), ("B", B, spb)):
        for split in ("train", "val", "test_chapman", "test_external", "test"):
            if (tag == "A" and split == "test") or (tag == "B" and split in ("test_chapman", "test_external")):
                continue
            s = "test_shdb" if (tag == "B" and split == "test") else split
            ids = sp.loc[sp.split == split, "idx"].to_numpy()
            offset = 0 if tag == "A" else len(np.load(A / "y.npy"))
            perm.extend((ids + offset).tolist())
    perm = np.array(perm, dtype=int)
    assert len(perm) == len(y), (len(perm), len(y))
    for key, (M, names) in {"expert": (e, en), "v2": (v, vn), "heat": (h, hn)}.items():
        np.save(OUT / f"features_{key}.npy", M[perm])
        pd.Series(names).to_csv(OUT / f"feature_names_{key}.csv", index=False, header=["name"])
    stats = {"n": len(y), "n_afib": int((y == 1).sum()), "n_afl": int((y == 0).sum()),
             "ratio": float((y == 0).sum() / (y == 1).sum()),
             "splits": {n: int(len(np.load(OUT / f"y_{n}.npy"))) for n in order},
             "sources": pd.concat(all_meta)["source"].value_counts().to_dict(),
             "features": {"expert": len(en), "v2": len(vn), "heat": len(hn)}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
