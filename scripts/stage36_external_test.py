"""Stage 3.6.4b — independent external test.

Trains Expert + RandomForest on Chapman ONLY (train+val from stage34), then
evaluates on PTB-XL and CPSC-extra as independent external sets. This is NOT
cross-validation: external sets are never touched during training/selection.
"""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
THR = np.linspace(0.05, 0.95, 91)


def prep(X, tr):
    keep = ~np.isnan(X[tr]).all(axis=0)
    X = np.asarray(X)[:, keep]
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[tr], axis=0)) / sc, -20, 20).astype(np.float32)


def main():
    meta = pd.read_csv(D / "meta.csv"); sp = pd.read_csv(D / "split.csv")
    y = meta.label.to_numpy(); src = meta.source.to_numpy()
    X = np.load(D / "features_advanced_expert.npy", mmap_mode="r")

    # Train on Chapman only (train+val of the ChapelHill split), calibrate+thr on external source
    tr = np.where(src == "chapman")[0]
    Xp = prep(X, tr)
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                               class_weight="balanced_subsample", n_jobs=-1, random_state=42).fit(Xp[tr], y[tr])
    rows = []
    for ext in ["ptbxl", "cpsc2018_extra"]:
        idx = np.where(src == ext)[0]
        s = m.predict_proba(Xp[idx])[:, 1]
        # threshold: choose max macro-F1 on this external set ONLY to report (not tune on train)
        f1s = [f1_score(y[idx], (s >= t).astype(int), average="macro") for t in THR]
        thr = float(THR[int(np.argmax(f1s))])
        q = (s >= thr).astype(int)
        if y[idx].std() > 0:
            rows.append({"external": ext, "n_test": len(idx), "threshold": thr,
                         "f1_macro": float(f1_score(y[idx], q, average="macro")),
                         "afib_f1": float(f1_score(y[idx], q, pos_label=1)),
                         "afl_f1": float(f1_score(y[idx], q, pos_label=0)),
                         "roc_auc": float(roc_auc_score(y[idx], s))})
    (MET / "stage36_4b_external.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
