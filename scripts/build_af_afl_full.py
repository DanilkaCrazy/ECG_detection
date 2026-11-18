"""Build a CLEAN full AF-vs-AFL binary dataset using ALL valid records.

Every AF/AFL record from the unified manifest is loaded, cleaned and unified to
a fixed geometry (500 Hz, 2500 samples by default, 12 leads).

- NaN -> 0, infinite -> clipped
- reject constant / clipped / unreadable signals
- resample to 500 Hz if needed, pad/trim to chosen length, pad/trim leads to 12
- group-aware 70/15/15 train/val/test split
- memory-safe chunked parallel WFDB loading into a memmap

Env: AF_AFL_OUT (dataset dir), AF_AFL_LEN (samples per record).
Outputs: X.npy, y.npy, X_{split}.npy, y_{split}.npy, meta.csv, split.csv,
         stats.json, qc_report.txt
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
OUT = config.PROCESSED_ROOT / os.environ.get("AF_AFL_OUT", "af_afl_full")
TARGET_FS = 500.0
TARGET_LEN = int(os.environ.get("AF_AFL_LEN", "2500"))
N_LEADS = 12
CHUNK = 800


def load_and_clean(row: dict) -> tuple[np.ndarray | None, dict]:
    import wfdb
    from scipy.signal import resample_poly

    qc = {"resampled": 0, "padded": 0, "truncated": 0, "nan_filled": 0}
    try:
        rec = wfdb.rdrecord(str(row["path_hea"]).replace(".hea", ""))
        sig = np.asarray(rec.p_signal, dtype=np.float32)
        if sig.size == 0:
            return None, qc
        if np.isnan(sig).any() or np.isinf(sig).any():
            qc["nan_filled"] = int(np.isnan(sig).sum() + np.isinf(sig).sum())
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        fs = float(rec.fs)
        if abs(fs - TARGET_FS) > 1e-6:
            g = gcd(int(TARGET_FS), int(fs))
            sig = resample_poly(sig, int(TARGET_FS) // g, int(fs) // g, axis=0).astype(np.float32)
            qc["resampled"] = 1
        if sig.shape[0] < TARGET_LEN:
            sig = np.pad(sig, ((0, TARGET_LEN - sig.shape[0]), (0, 0)), mode="constant")
            qc["padded"] = 1
        elif sig.shape[0] > TARGET_LEN:
            sig = sig[:TARGET_LEN]
            qc["truncated"] = 1
        if sig.shape[1] < N_LEADS:
            sig = np.pad(sig, ((0, 0), (0, N_LEADS - sig.shape[1])), mode="constant")
        elif sig.shape[1] > N_LEADS:
            sig = sig[:, :N_LEADS]
        s = float(np.std(sig))
        if s < 1e-3 or float(np.max(np.abs(sig))) > 100.0:
            return None, qc
        return np.ascontiguousarray(sig.T, dtype=np.float32), qc
    except Exception:
        return None, qc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.PROCESSED_ROOT / "unified_ecg_manifest.csv")
    sub = df[df["rhythm"].isin(["AFL", "AF"])].copy().reset_index(drop=True)
    n = len(sub)
    print(f"Found {n} AF/AFL records; loading in chunks of {CHUNK}...", flush=True)

    mm = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32,
                                   shape=(n, N_LEADS, TARGET_LEN))
    y = np.full(n, -1, dtype=np.int64)
    qc_totals = {"bad": 0, "resampled": 0, "padded": 0, "truncated": 0, "nan_filled": 0}
    meta_rows: list[dict | None] = [None] * n

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        rows = sub.iloc[start:end].to_dict("records")
        results = Parallel(n_jobs=4, verbose=0, batch_size=8)(
            delayed(load_and_clean)(r) for r in rows
        )
        for j, ((sig, qc), r) in enumerate(zip(results, rows)):
            gi = start + j
            for k in ("resampled", "padded", "truncated", "nan_filled"):
                qc_totals[k] += int(qc.get(k, 0) or 0)
            if sig is None:
                qc_totals["bad"] += 1
                continue
            mm[gi] = sig
            y[gi] = 1 if r["rhythm"] == "AF" else 0
            meta_rows[gi] = {
                "source": r["source"], "record_id": r["record_id"],
                "patient_id": r.get("patient_id"), "rhythm": r["rhythm"],
                "fs": r["fs"], "n_samp": r["n_samp"], "duration_s": r["duration_s"],
                "path_hea": r["path_hea"],
            }
        mm.flush()
        print(f"  {end}/{n}", flush=True)

    keep = np.where(y >= 0)[0]
    X = mm  # memmap; only `keep` rows are valid
    yk = y[keep]
    meta_df = pd.DataFrame([m for m in meta_rows if m is not None])

    grp = meta_df["patient_id"].fillna(meta_df["record_id"]).astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_local, tmp_local = next(gss.split(np.zeros((len(keep), 1)), yk, groups=grp))
    va_local, te_local = train_test_split(tmp_local, test_size=0.5,
                                          stratify=yk[tmp_local], random_state=42)
    tr, tmp = keep[tr_local], keep[tmp_local]
    va, te = keep[va_local], keep[te_local]

    np.save(OUT / "y.npy", y)
    for name, ids in [("train", tr), ("val", va), ("test", te)]:
        split_mm = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+",
                                             dtype=np.float32, shape=(len(ids), N_LEADS, TARGET_LEN))
        for i, gid in enumerate(ids):
            split_mm[i] = X[gid]
        split_mm.flush()
        np.save(OUT / f"y_{name}.npy", y[ids])
    meta_df.to_csv(OUT / "meta.csv", index=False)
    pd.DataFrame({
        "split": ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te),
        "idx": np.concatenate([tr, va, te]),
    }).to_csv(OUT / "split.csv", index=False)

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "AF_vs_AFL_binary_full",
        "n_input": int(n),
        "n_kept": int(len(keep)),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "n_af": int((yk == 1).sum()), "n_afl": int((yk == 0).sum()),
        "class_ratio_afl_af": float((yk == 0).sum() / max(int((yk == 1).sum()), 1)),
        "geometry": {"fs": TARGET_FS, "n_samp": TARGET_LEN, "n_leads": N_LEADS},
        "qc": qc_totals,
        "split": "group-aware (patient_id if available) 70/15/15",
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (OUT / "qc_report.txt").write_text("\n".join(f"{k}: {v}" for k, v in qc_totals.items()), encoding="utf-8")
    print("Done. Stats:", json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
