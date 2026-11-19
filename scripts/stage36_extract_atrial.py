"""Stage 3.6 P0 — correct atrial feature extraction on real ECG records.

Fixes Stage 3.5: QRS peaks are detected on (leads, samples); QRST residual is
passed into spectral and temporal analysis; atrial peaks are derived from the
residual (never from R-peaks); quality metrics are stored and unreliable
features remain NaN. NO model training or threshold selection happens here.
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal as sp_signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet import config
from ecgdet.atrial_signal_analysis import (
    multi_lead_qrs, spectral_atrial, multilead_coherence, _bandpass,
)
from ecgdet.atrial_residual import extract_atrial_residual

FS = 500.0
D = config.PROCESSED_ROOT / "af_afl_stage34"
OUT = ROOT / "data" / "processed" / "af_afl_stage36"
REP = ROOT / "reports" / "metrics" / "af_afl_stage36"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]
ATRIAL_LEADS = ["II", "III", "aVF", "V1"]


def atrial_peaks(resid, fs=FS):
    """Approximate F-wave cycle peaks from the atrial residual (NOT R-peaks)."""
    x = np.nan_to_num(resid)
    x = _bandpass(x, fs, 3, 12)
    if len(x) < 64:
        return np.array([], dtype=int)
    peaks, _ = sp_signal.find_peaks(
        x, distance=max(1, int(fs / 12.0)),
        prominence=max(float(np.std(x)) * 0.5, 1e-9))
    return peaks


def one(i, X):
    raw = np.asarray(X[i])  # (12, 5000) — correct orientation
    out = {}
    peaks, qinfo = multi_lead_qrs(raw, FS, LEADS)
    out["n_qrs_peaks"] = float(len(peaks))
    out["n_qrs_votes"] = float(qinfo.get("votes", 0))

    for method in ("median", "adaptive", "beat_aligned", "tq_segment"):
        r = extract_atrial_residual(raw, FS, LEADS, method=method, qrs_peaks=peaks)
        out[f"status_{method}"] = r["extraction_status"]
        out[f"qrs_supp_{method}"] = r["qrs_suppression_quality"]
        out[f"resid_ratio_{method}"] = r["residual_energy_ratio_composite"]
        out[f"resid_ratio_inside_{method}"] = r["residual_energy_ratio_inside_qrs"]
        if method == "tq_segment":
            for k in ("valid_tq_duration", "valid_tq_fraction", "n_valid_tq_segments",
                      "median_tq_segment_length"):
                out[f"{k}_tq"] = r[k]
        for lead in ATRIAL_LEADS:
            resid = r["residual"][LEADS.index(lead)]
            prefix = f"{method}_{lead}_"
            for k, v in spectral_atrial(resid, FS).items():
                out[prefix + k] = v
            if method != "tq_segment":
                from ecgdet.atrial_signal_analysis import temporal_atrial
                ap = atrial_peaks(resid, FS)
                for k, v in temporal_atrial(resid, FS, ap, qrs_peaks=peaks).items():
                    out[prefix + k] = v
        if method == "median":
            for k, v in multilead_coherence(r["residual"], FS, LEADS).items():
                out[k] = v
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    X = np.load(D / "X.npy", mmap_mode="r")
    meta = pd.read_csv(D / "meta.csv")
    res = Parallel(n_jobs=8, verbose=5, batch_size=6)(delayed(one)(i, X) for i in range(len(X)))
    keys = sorted({k for d in res for k in d})
    status_code = {"failed": 0.0, "degraded": 1.0, "ok": 2.0}

    def as_float(v):
        if isinstance(v, str):
            return status_code.get(v, np.nan)
        return float(v)

    M = np.asarray([[as_float(d.get(k, np.nan)) for k in keys] for d in res], dtype=np.float32)
    assert M.shape[0] == len(meta), (M.shape, len(meta))
    np.save(OUT / "features_atrial_residual.npy", M)
    pd.Series(keys).to_csv(OUT / "feature_names_atrial_residual.csv", index=False, header=["name"])

    keep_cols = [k for k in (res[0] if res else {}) if k.startswith("status_")
                 or k in ("n_qrs_peaks", "n_qrs_votes") or k.startswith("qrs_supp_")
                 or k.startswith("resid_ratio_") or k.startswith("valid_tq_")
                 or k in ("n_valid_tq_segments", "median_tq_segment_length")]
    diag_rows = []
    for i, d in enumerate(res):
        row = {"record_id": meta.record_id.iloc[i], "source": meta.source.iloc[i]}
        for k in keep_cols:
            row[k] = d.get(k, np.nan)
        diag_rows.append(row)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(REP / "stage36_extraction_diagnostics.csv", index=False)
    summary = {
        "n_records": len(meta),
        "n_features": M.shape[1],
        "qrs_peaks_median": float(np.nanmedian([d["n_qrs_peaks"] for d in res])),
        "n_zero_qrs": int(sum(d["n_qrs_peaks"] == 0 for d in res)),
        "status_counts": {m: str(diag[f"status_{m}"].value_counts().to_dict())
                          for m in ("median", "adaptive", "beat_aligned", "tq_segment")},
        "qrs_suppression_by_source": diag.groupby("source")["qrs_supp_median"].median().to_dict(),
        "residual_energy_ratio_inside_by_source": diag.groupby("source")["resid_ratio_inside_median"].median().to_dict(),
    }
    (REP / "stage36_extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
