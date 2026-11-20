"""Stage 3.6.22 — recompute features for ALL 7505 v1.1 records.

No dependency on the old 6180-row Stage 3.6.16 matrix: every feature array is
subset by `row_in_stage34` from the provenance-clean v1.1 dataset, and AV
features are recomputed on the full v1.1 row set.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402
from ecgdet.av_conduction_model import av_conduction_features, FEATURE_NAMES as AV_NAMES  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3614_v1_1"
S34 = ROOT / "data" / "processed" / "af_afl_stage34"
S36 = ROOT / "data" / "processed" / "af_afl_stage36"
S3610 = ROOT / "data" / "processed" / "af_afl_stage3610"
OUT = ROOT / "data" / "processed" / "af_afl_stage3622"
OUT.mkdir(parents=True, exist_ok=True)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def av_one(sig):
    peaks, _ = multi_lead_qrs(sig, 500.0, LEADS)
    rr = np.diff(peaks) / 500.0 if len(peaks) >= 2 else np.array([0.8])
    f = av_conduction_features(rr)
    return np.array([f[k] for k in AV_NAMES], dtype=np.float32)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    prov = pd.read_csv(D / "diagnosis_provenance.csv")
    rows = prov.row_in_stage34.to_numpy(dtype=int)

    expert = np.load(S34 / "features_advanced_expert.npy", mmap_mode="r")
    grouped = np.load(S3610 / "X.npy", mmap_mode="r")
    gn = pd.read_csv(S3610 / "feature_names.csv")
    Xraw = np.load(S34 / "X.npy", mmap_mode="r")[rows]
    AV = np.asarray(Parallel(n_jobs=8, batch_size=32, verbose=0)(
        delayed(av_one)(np.asarray(Xraw[i], dtype=np.float32)) for i in range(len(rows))),
        dtype=np.float32)

    X = np.column_stack([
        np.asarray(expert[rows], dtype=np.float32),
        np.asarray(grouped[rows], dtype=np.float32),
        AV,
    ]).astype(np.float32)
    names = ([f"X_expert_{i}" for i in range(expert.shape[1])]
             + gn.name.astype(str).tolist()
             + [f"AV_{n}" for n in AV_NAMES])
    groups = (["X"] * expert.shape[1] + gn.group.astype(str).tolist()
              + ["AV"] * len(AV_NAMES))
    assert X.shape[1] == len(names) == len(groups)

    np.save(OUT / "X.npy", X)
    pd.DataFrame({"name": names, "group": groups}).to_csv(OUT / "feature_names.csv", index=False)
    meta.to_csv(OUT / "meta.csv", index=False)
    pd.read_csv(D / "split.csv").to_csv(OUT / "split.csv", index=False)
    old_hash = pd.read_csv(S34 / "meta.csv").signal_hash.to_numpy()[rows]
    mismatch = int(np.sum(meta.signal_hash.to_numpy() != old_hash))
    stats = {
        "n_records": int(len(X)), "n_features": int(X.shape[1]),
        "by_group": {g: int(sum(x == g for x in groups)) for g in sorted(set(groups))},
        "source_rows": "all v1.1 records subset by row_in_stage34",
        "hash_mismatches": mismatch,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
