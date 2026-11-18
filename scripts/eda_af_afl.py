"""Exploratory data analysis of the unified ECG manifest and the AF/AFL subset.

Produces:
- reports/figures/eda/*.png  (class histograms, source/fs/duration/length distributions,
                             missing values, signal quality, AF-vs-AFL balance)
- reports/metrics/eda/data_analysis_report.txt  (human-readable summary)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures" / "eda"
MET = ROOT / "reports" / "metrics" / "eda"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

CLASS_ORDER = ["SB", "NSR", "AFL", "ST", "AF", "SA", "SVT", "VT"]
BINARY = ["AF", "AFL"]


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()


def _qc_one(row: dict) -> dict | None:
    """Quick per-record QC for one WFDB record (runs in a worker process)."""
    import wfdb
    from scipy.signal import resample_poly
    from math import gcd

    try:
        rec = wfdb.rdrecord(str(row["path_hea"]).replace(".hea", ""))
        sig = np.asarray(rec.p_signal, dtype=np.float32)
    except Exception:
        return None
    out: dict = {}
    if np.isnan(sig).any() or np.isinf(sig).any():
        out["nan"] = 1
    sig = np.nan_to_num(sig)
    if abs(float(rec.fs) - 500.0) > 1e-6:
        out["resampled"] = 1
        g = gcd(500, int(rec.fs))
        sig = resample_poly(sig, 500 // g, int(rec.fs) // g, axis=0).astype(np.float32)
    out["len"] = int(sig.shape[0])
    if sig.shape[0] < 2500:
        out["padded"] = 1
    elif sig.shape[0] > 2500:
        out["truncated"] = 1
    s = float(np.std(sig))
    out["std"] = s
    out["amp"] = float(np.max(np.abs(sig)))
    # WFDB signals are in mV: typical std 0.1-0.5
    if s < 1e-3 or out["amp"] > 100.0:
        out["bad_signal"] = 1
    else:
        out["ok"] = 1
    return out


def main() -> int:
    df = pd.read_csv(config.PROCESSED_ROOT / "unified_ecg_manifest.csv")
    lines: list[str] = []
    lines.append("ECG DATA ANALYSIS REPORT (unified manifest + AF/AFL subset)")
    lines.append("=" * 70)
    lines.append(f"Total records: {len(df)}")
    lines.append(f"Columns: {list(df.columns)}")

    # ---------------------------------------------------------------- 1. classes
    counts = df["rhythm"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    lines.append("")
    lines.append("1. CLASS DISTRIBUTION (all 8 classes):")
    for c, n in counts.items():
        lines.append(f"   {c:4s}: {n:6d}  ({100.0 * n / len(df):5.2f}%)")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", fontsize=8)
    plt.title("Class distribution — unified dataset (44,075 records)")
    plt.ylabel("records")
    plt.ylim(0, counts.max() * 1.18)
    savefig("class_distribution_all.png")

    # AF vs AFL
    sub = df[df["rhythm"].isin(BINARY)].copy().reset_index(drop=True)
    af_n = int((sub["rhythm"] == "AF").sum())
    afl_n = int((sub["rhythm"] == "AFL").sum())
    ratio = afl_n / max(af_n, 1)
    lines.append("")
    lines.append(f"2. AF vs AFL SUBSET: AF={af_n}, AFL={afl_n}, ratio AFL/AF={ratio:.2f}")
    lines.append("   (the previous binary model used only 300+300 = 600 of these records!)")

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=["AF", "AFL"], y=[af_n, afl_n], palette=["#e74c3c", "#3498db"])
    for i, v in enumerate([af_n, afl_n]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.title(f"AF vs AFL imbalance (ratio {ratio:.2f} : 1)")
    plt.ylabel("records")
    savefig("af_afl_imbalance.png")

    # per source
    src = pd.crosstab(sub["source"], sub["rhythm"]).reindex(columns=BINARY, fill_value=0)
    src["total"] = src.sum(axis=1)
    src = src.sort_values("total", ascending=False)
    lines.append("")
    lines.append("3. AF/AFL BY SOURCE:")
    lines.append(src.to_string())

    plt.figure(figsize=(10, 5))
    src[BINARY].plot(kind="bar", stacked=True, color=["#e74c3c", "#3498db"])
    plt.title("AF / AFL records by source dataset")
    plt.ylabel("records")
    plt.xticks(rotation=45, ha="right")
    savefig("af_afl_by_source.png")

    # full per-source class table
    src_all = pd.crosstab(df["source"], df["rhythm"])
    lines.append("")
    lines.append("4. ALL CLASSES BY SOURCE:")
    lines.append(src_all.to_string())

    # ---------------------------------------------------------------- formats
    lines.append("")
    lines.append("5. FORMAT / FREQUENCY / LENGTH:")
    for col in ["fs", "n_samp", "duration_s"]:
        lines.append(f"   {col}: min={df[col].min()}, max={df[col].max()}, "
                     f"median={df[col].median()}, unique={df[col].nunique()}")
    lines.append(f"   fs values: {sorted(df['fs'].unique())[:20]}")
    lines.append(f"   'leads' unique count: {df['leads'].nunique()}")
    lines.append(f"   patient_id missing: {int(df['patient_id'].isna().sum())} / {len(df)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.histplot(df["fs"], bins=30, ax=axes[0, 0], kde=False)
    axes[0, 0].set_title("Sampling frequency (fs)")
    sns.histplot(df["duration_s"], bins=50, ax=axes[0, 1])
    axes[0, 1].set_title("Record duration (s)")
    sns.histplot(df["n_samp"], bins=50, ax=axes[1, 0])
    axes[1, 0].set_title("Number of samples")
    sns.countplot(y=df["source"], ax=axes[1, 1], order=df["source"].value_counts().index)
    axes[1, 1].set_title("Records per source")
    savefig("format_distributions.png")

    # fs/duration per class for AF/AFL
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=sub, x="rhythm", y="fs", ax=axes[0])
    axes[0].set_title("Sampling frequency: AF vs AFL")
    sns.boxplot(data=sub, x="rhythm", y="duration_s", ax=axes[1])
    axes[1].set_title("Duration: AF vs AFL")
    savefig("af_afl_fs_duration.png")

    # ---------------------------------------------------------------- missing
    na = df.isna().sum()
    lines.append("")
    lines.append("6. MISSING VALUES PER COLUMN:")
    for c, v in na.items():
        lines.append(f"   {c:12s}: {v:6d} ({100*v/len(df):5.2f}%)")

    plt.figure(figsize=(10, 4))
    sns.barplot(x=na.index, y=na.values, palette="rocket")
    plt.title("Missing values per column (unified manifest)")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    savefig("missing_values.png")

    # ---------------------------------------------------------------- signal QC
    lines.append("")
    lines.append("7. SIGNAL QUALITY CHECK (random sample of 600 AF/AFL records):")
    from joblib import Parallel, delayed

    qc = {"nan": 0, "bad_path": 0, "resampled": 0, "padded": 0,
          "truncated": 0, "ok": 0, "bad_signal": 0}
    sample = sub.sample(n=min(600, len(sub)), random_state=42).to_dict("records")
    results = Parallel(n_jobs=8, verbose=0)(delayed(_qc_one)(r) for r in sample)
    amps, stds, lengths = [], [], []
    bad_examples: list[str] = []
    for r, out in zip(sample, results):
        if out is None:
            qc["bad_path"] += 1
            if len(bad_examples) < 10:
                bad_examples.append(str(r["record_id"]))
            continue
        for k in qc:
            qc[k] += int(out.get(k, 0))
        if out.get("std") is not None:
            stds.append(out["std"])
            amps.append(out["amp"])
            lengths.append(out["len"])
    for k, v in qc.items():
        lines.append(f"   {k:10s}: {v}")
    if bad_examples:
        lines.append(f"   bad path examples: {bad_examples}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(stds, bins=60, ax=axes[0])
    axes[0].set_title("Signal std (µV) — AF/AFL")
    sns.histplot(amps, bins=60, ax=axes[1])
    axes[1].set_title("Max |amplitude| (µV)")
    sns.histplot(lengths, bins=60, ax=axes[2])
    axes[2].set_title("Samples before trim/pad")
    savefig("signal_quality.png")

    plt.figure(figsize=(9, 5))
    names = list(qc.keys())
    vals = [qc[k] for k in names]
    ax = sns.barplot(x=names, y=vals, palette="muted")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.title("Signal QC summary (AF/AFL)")
    plt.xticks(rotation=45, ha="right")
    savefig("signal_qc_summary.png")

    # ---------------------------------------------------------------- conclusions
    lines.append("")
    lines.append("8. CONCLUSIONS / RECOMMENDATIONS:")
    lines.append("   - AF/AFL ratio in raw data is ~4.4:1; the 300/300 subsample discarded ~87% of data.")
    lines.append("   - Rebuild the binary dataset using ALL valid AF+AFL records.")
    lines.append("   - Apply imbalance handling: class weights, oversampling, undersampling, SMOTE,")
    lines.append("     threshold tuning, focal loss; compare all on val/test.")
    lines.append("   - Unify geometry: resample to 500 Hz, pad/trim to 2500 samples, 12 leads,")
    lines.append("     NaN -> 0, drop constant/clipped signals.")
    lines.append("   - Extend features: spectral bands, peak morphology, RR irregularity, f-wave proxies.")

    (MET / "data_analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]), flush=True)
    print("EDA artifacts saved to", FIG, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
