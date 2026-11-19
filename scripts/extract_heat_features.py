"""Extract HEAT-inspired AV-conduction features for an AF/AFL dataset."""

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
from ecgdet.expert_af_afl import _detect_peaks
from ecgdet.heat_inspired import heat_features

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / os.environ.get("AF_AFL_OUT", "af_afl_full10s_dedup")


def one(i: int, X) -> dict:
    sig = np.asarray(X[i], dtype=np.float32)
    peaks = _detect_peaks(sig[1] if sig.shape[0] > 1 else sig[0], 500.0)
    rr = np.diff(peaks) / 500.0 * 1000.0
    return heat_features(rr)


def main() -> int:
    X = np.load(DATA / "X.npy", mmap_mode="r")
    n = len(X)
    print(f"HEAT features for {n} records", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=8, verbose=5, batch_size=16)(delayed(one)(i, X) for i in range(n))
    keys = sorted(res[0].keys())
    M = np.array([[float(d.get(k, np.nan)) for k in keys] for d in res], dtype=np.float32)
    np.save(DATA / "features_heat.npy", M)
    pd.Series(keys).to_csv(DATA / "feature_names_heat.csv", index=False, header=["name"])
    print(f"Done in {time.time()-t0:.1f}s: {M.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
