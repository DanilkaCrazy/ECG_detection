"""Merge Chapman STRICT + external AF datasets into a balanced training set.

Splits: Chapman strict test stays an independent in-domain test; external test
is kept as a separate out-of-domain holdout. Train/val contain both sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
CH = config.PROCESSED_ROOT / "af_afl_strict"
EXT = config.PROCESSED_ROOT / "af_external"
OUT = config.PROCESSED_ROOT / "af_afl_merged"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    chm = pd.read_csv(CH / "meta.csv")
    exm = pd.read_csv(EXT / "meta.csv")
    chy = np.load(CH / "y.npy"); exy = np.load(EXT / "y.npy")
    csp = pd.read_csv(CH / "split.csv"); esp = pd.read_csv(EXT / "split.csv")
    # strict dataset has only split files -> rebuild global memmap
    CX = np.lib.format.open_memmap(CH / "X.npy", mode="w+", dtype=np.float32, shape=(len(chy), 12, 5000))
    for split in ("train", "val", "test"):
        ids = csp.loc[csp.split == split, "idx"].to_numpy()
        a = np.load(CH / f"X_{split}.npy", mmap_mode="r")
        CX[ids] = a
    CX.flush()
    EX = np.load(EXT / "X.npy", mmap_mode="r")

    parts = {"train": [], "val": [], "test_chapman": [], "test_external": []}
    for tag, meta, y, sp, X in (("chapman", chm, chy, csp, CX), ("external", exm, exy, esp, EX)):
        for split in ("train", "val", "test"):
            ids = sp.loc[sp.split == split, "idx"].to_numpy()
            out_name = split if split != "test" else f"test_{tag}"
            parts[out_name].append((meta.iloc[ids].reset_index(drop=True), y[ids], X, ids))

    all_meta, all_y = [], []
    for name, entries in parts.items():
        n = sum(len(e[1]) for e in entries)
        dst = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+",
                                        dtype=np.float32, shape=(n, 12, 5000))
        pos = 0
        rows = []
        for meta, yy, X, ids in entries:
            for i in ids:
                dst[pos] = X[i]; pos += 1
            for j, (_, r) in enumerate(meta.iterrows()):
                d = r.to_dict(); d["label"] = int(yy[j]); d["split"] = name
                rows.append(d)
            all_y.append(yy)
        dst.flush()
        pd.DataFrame(rows).to_csv(OUT / f"meta_{name}.csv", index=False)
        np.save(OUT / f"y_{name}.npy", np.concatenate([e[1] for e in entries]))
        print(name, n, "AFIB", int((np.concatenate([e[1] for e in entries]) == 1).sum()), flush=True)
        all_meta.extend(rows)

    y = np.concatenate(all_y)
    np.save(OUT / "y.npy", y)
    pd.DataFrame(all_meta).to_csv(OUT / "meta.csv", index=False)
    # global X for feature extraction: train+val+test_chapman+test_external in that order
    order = ["train", "val", "test_chapman", "test_external"]
    ns = [len(np.load(OUT / f"y_{n}.npy")) for n in order]
    gx = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32,
                                   shape=(sum(ns), 12, 5000))
    pos = 0
    for n in order:
        a = np.load(OUT / f"X_{n}.npy", mmap_mode="r")
        gx[pos:pos + len(a)] = a; pos += len(a)
    gx.flush()
    split_rows = []
    for n in order:
        split_rows.extend([n] * len(np.load(OUT / f"y_{n}.npy")))
    pd.DataFrame({"split": split_rows, "idx": np.arange(sum(ns))}).to_csv(OUT / "split.csv", index=False)
    stats = {"n": int(len(y)), "n_afib": int((y == 1).sum()), "n_afl": int((y == 0).sum()),
             "ratio": float((y == 0).sum() / (y == 1).sum()),
             "splits": {n: int(len(np.load(OUT / f"y_{n}.npy"))) for n in order},
             "sources": pd.DataFrame(all_meta)["source"].value_counts().to_dict()}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
