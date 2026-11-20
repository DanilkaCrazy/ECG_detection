"""Stage 3.6.9 (P0) — QRS detector audit after the 8-18 Hz / tachycardia fixes.

* annotated WFDB records with tolerances expressed in MILLISECONDS;
* synthetic edge cases: short RR, single lead, missing chest leads, F-waves;
* per-dataset sensitivity, PPV, F1 and median localisation error.
"""
from __future__ import annotations

import json
import sys
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import wfdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402
from qrs_detection_benchmark import match, QRS_SYMS  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "metrics" / "stage369"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["mitbih", "edb", "qtdb", "svdb"]
TOL_MS = (25, 50, 100)
MAX_RECORDS = 40
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def synthetic_cases():
    rows = []
    fs = 500.0
    cases = (("normal", 1000, 0), ("short_rr", 300, 0),
             ("tachy_240ms", 240, 0), ("f_waves", 500, 0.25),
             ("single_lead", 500, 0), ("missing_chest", 500, 0))
    for name, rr_ms, f_wave in cases:
        rr = int(rr_ms / 1000 * fs)
        x = np.zeros((12, int(rr * 42)))
        ref = np.arange(50, x.shape[1] - 50, rr)
        tmpl = np.array([-.1, -.2, 1.0, -.3, -.1, 0, 0, 0, .05])
        for p in ref:
            for k in range(12):
                x[k, p - 3:p + 6] += tmpl
        if f_wave:
            x += f_wave * np.sin(2 * np.pi * 5 * np.arange(x.shape[1]) / fs)
        if name == "single_lead":
            x[1:] = 0
        if name == "missing_chest":
            x[6:] = 0
        found, info = multi_lead_qrs(x, fs, LEADS)
        tp, fp, fn, mae, _ = match(found, ref, int(0.05 * fs))
        rows.append({"case": name, "n_ref": len(ref), "n_found": len(found),
                     "tp": tp, "fp": fp, "fn": fn,
                     "recall": tp / len(ref), "precision": tp / max(len(found), 1),
                     "status": info["qrs_detection_status"]})
    return pd.DataFrame(rows)


def annotated_cases():
    rows = []
    for ds in DATASETS:
        for f in sorted((RAW / ds).glob("**/*.atr"))[:MAX_RECORDS]:
            name = str(f).replace("\\", "/").removesuffix(".atr")
            if not pathlib.Path(name + ".hea").exists():
                continue
            try:
                rec = wfdb.rdrecord(name)
                ann = wfdb.rdann(name, "atr")
            except Exception:
                continue
            ref = ann.sample[np.isin(ann.symbol, list(QRS_SYMS))]
            sig = np.asarray(rec.p_signal, dtype=float)
            if sig.ndim == 1:
                sig = sig[:, None]
            if sig.shape[1] == len(rec.sig_name) and sig.shape[0] != len(rec.sig_name):
                sig = sig.T
            found, info = multi_lead_qrs(sig, float(rec.fs), list(rec.sig_name))
            for tol_ms in TOL_MS:
                tol = int(round(tol_ms / 1000.0 * rec.fs))
                tp, fp, fn, mae, _ = match(found, ref, tol)
                rows.append({"dataset": ds, "record": f.stem, "fs": rec.fs,
                             "tol_ms": tol_ms, "tol_samples": tol,
                             "n_ref": len(ref), "n_found": len(found),
                             "tp": tp, "fp": fp, "fn": fn,
                             "mae_ms": mae * 1000 / rec.fs if mae else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    syn = synthetic_cases()
    syn.to_csv(OUT / "qrs_edge_cases.csv", index=False)
    ann = annotated_cases()
    ann.to_csv(OUT / "qrs_annotated_by_tolerance.csv", index=False)
    agg = ann.groupby(["dataset", "tol_ms"]).agg(
        n_ref=("n_ref", "sum"), n_found=("n_found", "sum"), tp=("tp", "sum"),
        fp=("fp", "sum"), fn=("fn", "sum"), mae_ms=("mae_ms", "median")).reset_index()
    agg["sensitivity"] = agg.tp / agg.n_ref
    agg["ppv"] = agg.tp / agg.n_found
    agg["f1"] = 2 * agg.tp / (2 * agg.tp + agg.fp + agg.fn)
    agg.to_csv(OUT / "qrs_annotated_summary.csv", index=False)
    report = {
        "note": "Tolerances in milliseconds. 8-18 Hz bandpass and adaptive refractory after Stage 3.6.9 audit.",
        "synthetic": syn.to_dict("records"),
        "annotated": agg.to_dict("records"),
    }
    (OUT / "qrs_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(agg.round(4).to_string(index=False))
    print(syn.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
