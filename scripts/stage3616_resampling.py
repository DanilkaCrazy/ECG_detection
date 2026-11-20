"""Stage 3.6.16 H5 — compare 125 / 250 / 500 Hz with correct anti-alias resampling.

A bounded, stratified subset is used so the experiment stays CPU-feasible.
The SAME records and split are used for every rate; only the sampling rate
changes.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from sklearn.ensemble import RandomForestClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.dataset_harmonization import decimate_signal  # noqa: E402
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402
from ecgdet.rhythm_features import rhythm_features, FEATURE_NAMES as RR_NAMES  # noqa: E402
from ecgdet.external_evaluation import evaluate_predictions, select_threshold  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3614"
SRC = ROOT / "data" / "processed" / "af_afl_stage34" / "X.npy"
MET = ROOT / "reports" / "metrics" / "stage3616"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]
RATES = [500, 250, 125]
MAX_PER_GROUP = 120


def features(sig, fs):
    x = np.asarray(sig, dtype=float)
    out = []
    # Per-lead band power and amplitude descriptors.
    for lead in x:
        f, p = sp_signal.welch(lead - lead.mean(), fs, nperseg=min(1024, len(lead)))
        tot = float(p.sum()) + 1e-12
        for lo, hi in ((0.5, 4), (4, 8), (8, 16), (16, 32), (32, min(60, fs / 2 - 1))):
            band = (f >= lo) & (f < hi)
            out.append(float(p[band].sum() / tot))
        out += [float(np.std(lead)), float(np.percentile(np.abs(lead), 95))]
    peaks, _ = multi_lead_qrs(x, fs, LEADS)
    rr = np.diff(peaks) / fs if len(peaks) >= 2 else np.array([0.8])
    r = rhythm_features(rr)
    out += [r[k] for k in RR_NAMES]
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    prov = pd.read_csv(D / "diagnosis_provenance.csv")
    rows = prov.row_in_stage34.to_numpy(dtype=int)
    rng = np.random.default_rng(42)
    pick = []
    for (s, l), g in meta.groupby(["source", "label"]):
        idx = g.index.to_numpy()
        pick.extend(rng.choice(idx, size=min(MAX_PER_GROUP, len(idx)), replace=False).tolist())
    pick = np.sort(np.asarray(pick, dtype=int))
    Xraw = np.load(SRC, mmap_mode="r")[rows[pick]]
    sub = meta.iloc[pick].reset_index(drop=True)

    fit = np.where((sub.source == "chapman") & (sub.split == "train"))[0]
    val = np.where((sub.source == "chapman") & (sub.split == "val"))[0]
    test = np.where((sub.source == "chapman") & (sub.split == "test"))[0]

    results = []
    for fs in RATES:
        rate = fs / 500.0
        F = np.asarray([features(decimate_signal(Xraw[i], 500.0, fs) if rate < 1
                                 else Xraw[i], fs) for i in range(len(sub))], dtype=np.float32)
        med = np.nan_to_num(np.nanmedian(F[fit], axis=0))
        F = np.where(np.isnan(F), med, F)
        q1, q3 = np.percentile(F[fit], [25, 75], axis=0)
        sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
        F = np.clip((F - np.median(F[fit], axis=0)) / sc, -20, 20).astype(np.float32)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=42).fit(F[fit], sub.label.to_numpy()[fit])
        s_val = m.predict_proba(F[val])[:, 1]
        thr = select_threshold(sub.label.to_numpy()[val], s_val)
        met = evaluate_predictions(sub.label.to_numpy()[test], m.predict_proba(F[test])[:, 1], thr)
        results.append({"fs": fs, "n": len(sub), "n_features": F.shape[1],
                        "f1_macro": met["f1_macro"], "afib_f1": met["afib_f1"],
                        "afl_f1": met["afl_f1"], "roc_auc": met["roc_auc"],
                        "pr_auc": met["pr_auc"]})
        print(fs, "Hz F1", round(met["f1_macro"], 4), "AUC", round(met["roc_auc"], 4), flush=True)

    df = pd.DataFrame(results)
    df.to_csv(MET / "resampling_comparison.csv", index=False)
    (MET / "resampling_comparison.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
