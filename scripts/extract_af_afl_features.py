"""Batch-extract advanced AF/AFL features (cached to disk)."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / os.environ.get("AF_AFL_OUT", "af_afl_full")
FEAT_VERSION = os.environ.get("AF_AFL_FEAT", "v1")
if FEAT_VERSION == "v2":
    from ecgdet.features_advanced_v2 import extract_v2_features as extract_fn
elif FEAT_VERSION == "expert":
    from ecgdet.expert_af_afl import extract_expert_features as extract_fn
else:
    from ecgdet.features_advanced import extract_advanced_features as extract_fn


def one(i: int) -> tuple[int, dict]:
    X = np.load(DATA / "X.npy", mmap_mode="r")
    sig2d = np.asarray(X[i], dtype=np.float64).T  # (2500, 12)
    try:
        return i, extract_fn(sig2d, fs=500.0)
    except Exception:
        return i, {}


def main() -> int:
    n = len(np.load(DATA / "y.npy"))
    print(f"Extracting advanced features for {n} records...", flush=True)
    t0 = time.time()
    n_jobs = int(os.environ.get("AF_AFL_JOBS", "8"))
    results = Parallel(n_jobs=n_jobs, verbose=5, batch_size=8)(delayed(one)(i) for i in range(n))
    results.sort(key=lambda t: t[0])
    keys = sorted({k for _, d in results for k in d.keys()})
    key_pos = {k: j for j, k in enumerate(keys)}
    M = np.full((n, len(keys)), np.nan, dtype=np.float32)
    for i, d in results:
        for k, v in d.items():
            if v is None or k not in key_pos:
                continue
            try:
                M[i, key_pos[k]] = float(v)
            except (ValueError, TypeError):
                pass
    suffix = "" if FEAT_VERSION == "v1" else f"_{FEAT_VERSION}"
    np.save(DATA / f"features_advanced{suffix}.npy", M)
    pd.Series(keys).to_csv(DATA / f"feature_names{suffix}.csv", index=False, header=["name"])
    print(f"Done in {time.time()-t0:.1f}s. Shape={M.shape}, NaNs={np.isnan(M).sum()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
