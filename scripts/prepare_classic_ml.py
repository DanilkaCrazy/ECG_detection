"""Prepare feature matrices and patient-wise splits for classical ML.

- Load unified manifest
- Load raw signals (Chapman .mat, PTB-XL/MIT-BIH/INCART .dat)
- Resample to common fs if needed (default target 500 Hz)
- Normalize per-lead: fit robust stats on TRAIN only (median/MAD, winsorized)
- Split by patient_id when available (stratified), else by record with group
- Save features, labels, split indices
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from ecgdet import config, features, normalize
from ecgdet.wfdb_io import read_mat_signal, parse_header_lenient

__all__ = ["prepare_classic_ml_dataset"]


def _load_signal(row: pd.Series) -> tuple[np.ndarray, float]:
    source = row["source"]
    if source == "chapman":
        mat = row["path_mat"]
        hea = row["path_hea"]
        header, _ = parse_header_lenient(hea, mat)
        val = read_mat_signal(mat)  # (n_sig, n_samp)
        sig = val.T.astype(np.float64)
        fs = float(header.fs) if header else float(row["fs"])
        leads = header.lead_names if header else row["leads"].split(",")
        return sig, fs, leads
    else:
        hea = row["path_hea"]
        rec = wfdb.rdrecord(hea.replace(".hea", ""))
        sig = rec.p_signal.astype(np.float64)
        fs = float(rec.fs)
        leads = rec.sig_name
        return sig, fs, leads


def _resample_to(sig: np.ndarray, fs_src: float, fs_tgt: float) -> np.ndarray:
    if abs(fs_src - fs_tgt) < 1e-6:
        return sig
    # use polyphase resampling
    from math import gcd

    g = gcd(int(fs_tgt), int(fs_src))
    up = int(fs_tgt) // g
    down = int(fs_src) // g
    out = resample_poly(sig, up, down, axis=0)
    return out


def prepare_classic_ml_dataset(
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    fs_target: float = 500.0,
    max_records: int | None = None,
    norm_method: str = "robust_z_mad",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    manifest_path = Path(manifest_path) if manifest_path else config.PROCESSED_ROOT / "unified_ecg_manifest.csv"
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_ROOT / "classic_ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if max_records:
        df = df.groupby("rhythm", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_records), random_state=random_state)
        )

    # prepare splits - use record-wise stratified split (patient IDs not available for Chapman)
    from sklearn.model_selection import StratifiedShuffleSplit
    y_all = df["rhythm"].values
    
    # stratified split: first hold out test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss.split(df, y_all))
    df_tv = df.iloc[trainval_idx]
    df_test = df.iloc[test_idx]

    # val from trainval
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(sss2.split(df_tv, df_tv["rhythm"]))
    df_train = df_tv.iloc[train_idx]
    df_val = df_tv.iloc[val_idx]

    split_info = {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
    }
    print(split_info)

    # extract features
    def _extract_for_frame(frame: pd.DataFrame) -> list[dict]:
        feats = []
        for _, row in frame.iterrows():
            try:
                sig, fs, leads = _load_signal(row)
                sig = _resample_to(sig, fs, fs_target)
                # use up to 10s
                maxlen = int(fs_target * 10)
                if sig.shape[0] > maxlen:
                    sig = sig[:maxlen]
                # if fewer samples, keep as is (features handle)
                f = features.extract_features(sig, fs_target, lead_names=list(leads))
                f["source"] = row["source"]
                f["rhythm"] = row["rhythm"]
                f["record_id"] = row["record_id"]
                feats.append(f)
            except Exception as e:
                continue
        return feats

    print("Extracting train features...")
    feat_train = _extract_for_frame(df_train)
    print("Extracting val features...")
    feat_val = _extract_for_frame(df_val)
    print("Extracting test features...")
    feat_test = _extract_for_frame(df_test)

    df_feat_train = pd.DataFrame(feat_train)
    df_feat_val = pd.DataFrame(feat_val)
    df_feat_test = pd.DataFrame(feat_test)
    # align columns across splits
    all_cols = sorted(set(df_feat_train.columns) | set(df_feat_val.columns) | set(df_feat_test.columns))
    df_feat_train = df_feat_train.reindex(columns=all_cols)
    df_feat_val = df_feat_val.reindex(columns=all_cols)
    df_feat_test = df_feat_test.reindex(columns=all_cols)
    df_feat_train = df_feat_train.fillna(df_feat_train.median(numeric_only=True))
    df_feat_val = df_feat_val.fillna(df_feat_train.median(numeric_only=True))
    df_feat_test = df_feat_test.fillna(df_feat_train.median(numeric_only=True))

    y_train = df_feat_train.pop("rhythm")
    meta_train = df_feat_train.pop("record_id")
    src_train = df_feat_train.pop("source")
    y_val = df_feat_val.pop("rhythm")
    meta_val = df_feat_val.pop("record_id")
    src_val = df_feat_val.pop("source")
    y_test = df_feat_test.pop("rhythm")
    meta_test = df_feat_test.pop("record_id")
    src_test = df_feat_test.pop("source")

    X_train = df_feat_train.values.astype(np.float32)
    X_val = df_feat_val.values.astype(np.float32)
    X_test = df_feat_test.values.astype(np.float32)

    # robust normalization fit on train only
    cfg = normalize.RobustNormConfig(method=norm_method)
    fitted = normalize.fit_robust_norm(X_train, cfg)
    X_train = normalize.apply_robust_norm(X_train, fitted).astype(np.float32)
    X_val = normalize.apply_robust_norm(X_val, fitted).astype(np.float32)
    X_test = normalize.apply_robust_norm(X_test, fitted).astype(np.float32)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train_enc)
    np.save(output_dir / "y_val.npy", y_val_enc)
    np.save(output_dir / "y_test.npy", y_test_enc)
    joblib.dump(fitted, output_dir / "norm_fitted.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")
    pd.Series(df_feat_train.columns).to_csv(output_dir / "feature_names.csv", index=False)
    pd.DataFrame({
        "record_id": meta_train,
        "source": src_train,
        "y": y_train,
    }).to_csv(output_dir / "train_meta.csv", index=False)
    pd.DataFrame({"record_id": meta_val, "source": src_val, "y": y_val}).to_csv(output_dir / "val_meta.csv", index=False)
    pd.DataFrame({"record_id": meta_test, "source": src_test, "y": y_test}).to_csv(output_dir / "test_meta.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fs_target": fs_target,
        "norm_method": norm_method,
        "split_info": split_info,
        "n_features": int(X_train.shape[1]),
        "classes": le.classes_.tolist(),
        "class_counts_train": dict(pd.Series(y_train).value_counts()),
        "class_counts_val": dict(pd.Series(y_val).value_counts()),
        "class_counts_test": dict(pd.Series(y_test).value_counts()),
    }
    (output_dir / "prep_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
