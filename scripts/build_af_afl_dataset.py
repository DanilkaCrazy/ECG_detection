"""Build the binary AF-vs-AFL sub-dataset from the unified ECG manifest.

Task (sub-task of the ELTA arrhythmia project):
    binary recognition of AF (atrial fibrillation) vs AFL (atrial flutter)
    on raw 12-lead ECG signals, to later be inserted into the overall 8-class
    arrhythmia model as a specialised head / cascade stage.

Inputs
------
- ``data/processed/unified_ecg_manifest.csv`` (built by ``build_unified_dataset``)

Outputs
-------
- ``data/processed/af_afl/X.npy``       (N, n_leads, n_samp) float32
- ``data/processed/af_afl/y.npy``       (N,) int64          (AFL=0, AF=1)
- ``data/processed/af_afl/meta.csv``   per-record metadata
- ``data/processed/af_afl/split.csv``  train/val/test indices
- ``data/processed/af_afl/stats.json`` subsample description
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = config.PROCESSED_ROOT / "af_afl"

TARGET_FS = 500.0
TARGET_LEN = 2500  # 5 seconds @ 500 Hz
N_LEADS = 12


def load_raw_signal(row: pd.Series) -> np.ndarray | None:
    """Load one record -> (n_leads, n_samp) float32 at fixed geometry."""
    import wfdb
    from scipy.signal import resample_poly
    from math import gcd

    try:
        rec = wfdb.rdrecord(str(row["path_hea"]).replace(".hea", ""))
        sig = rec.p_signal.astype(np.float32)
        fs = rec.fs
        if abs(fs - TARGET_FS) > 1e-6:
            g = gcd(TARGET_FS, int(fs))
            up = int(TARGET_FS // g)
            down = int(fs // g)
            sig = resample_poly(sig, up, down, axis=0).astype(np.float32)
        if sig.shape[0] > TARGET_LEN:
            sig = sig[:TARGET_LEN]
        elif sig.shape[0] < TARGET_LEN:
            pad = TARGET_LEN - sig.shape[0]
            sig = np.pad(sig, ((0, pad), (0, 0)), mode="constant")
        if sig.shape[1] < N_LEADS:
            sig = np.pad(sig, ((0, 0), (0, N_LEADS - sig.shape[1])), mode="constant")
        elif sig.shape[1] > N_LEADS:
            sig = sig[:, :N_LEADS]
        sig = np.nan_to_num(sig, nan=0.0)
        # -> (n_leads, n_samp)
        return np.transpose(sig, (1, 0)).astype(np.float32)
    except Exception:
        return None


def main(max_per_class: int = 300) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.PROCESSED_ROOT / "unified_ecg_manifest.csv")
    sub = df[df["rhythm"].isin(["AF", "AFL"])].copy()

    kept_rows = []
    for rhythm, cap in [("AF", max_per_class), ("AFL", max_per_class)]:
        if rhythm not in sub["rhythm"].values:
            continue
        part = sub[sub["rhythm"] == rhythm]
        n = min(len(part), cap)
        kept_rows.append(part.sample(n=n, random_state=42))
    df_bin = pd.concat(kept_rows, ignore_index=True).reset_index(drop=True)

    X_list, y_list, meta = [], [], []
    for _, row in df_bin.iterrows():
        sig = load_raw_signal(row)
        if sig is None:
            continue
        X_list.append(sig)
        y_list.append(1 if row["rhythm"] == "AF" else 0)  # AF=1, AFL=0
        meta.append(
            {
                "source": row["source"],
                "record_id": row["record_id"],
                "rhythm": row["rhythm"],
                "fs": row["fs"],
                "n_samp": row["n_samp"],
                "duration_s": row["duration_s"],
                "leads": row["leads"],
                "path_hea": row["path_hea"],
            }
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta)

    from sklearn.model_selection import train_test_split

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42
    )

    np.save(OUT_DIR / "X.npy", X.astype(np.float32))
    np.save(OUT_DIR / "y.npy", y.astype(np.int64))
    np.save(OUT_DIR / "X_train.npy", X_train.astype(np.float32))
    np.save(OUT_DIR / "X_val.npy", X_val.astype(np.float32))
    np.save(OUT_DIR / "X_test.npy", X_test.astype(np.float32))
    np.save(OUT_DIR / "y_train.npy", y_train.astype(np.int64))
    np.save(OUT_DIR / "y_val.npy", y_val.astype(np.int64))
    np.save(OUT_DIR / "y_test.npy", y_test.astype(np.int64))
    meta_df.to_csv(OUT_DIR / "meta.csv", index=False)

    split_rows = pd.DataFrame(
        {
            "split": ["train"] * len(X_train) + ["val"] * len(X_val) + ["test"] * len(X_test),
            "idx": list(range(len(X_train))) + list(range(len(X_train), len(X_train) + len(X_val))) + list(range(len(X_train) + len(X_val), len(X))),
        }
    )
    split_rows.to_csv(OUT_DIR / "split.csv", index=False)

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "AF_vs_AFL_binary",
        "n_records_total": int(len(df_bin)),
        "n_kept": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "max_per_class": int(max_per_class),
        "n_af_total_in_unified": int((df["rhythm"] == "AF").sum()),
        "n_afl_total_in_unified": int((df["rhythm"] == "AFL").sum()),
        "binary_label_encoding": {"AFL": 0, "AF": 1},
        "signal_geometry": {"fs": TARGET_FS, "n_samp": TARGET_LEN, "n_leads": N_LEADS},
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"AF-vs-AFL binary dataset: kept={len(X)} train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"  y counts: AF={int((y == 1).sum())} AFL={int((y == 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(max_per_class=300))
