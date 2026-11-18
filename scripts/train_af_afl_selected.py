"""Feature-selection experiment for AF vs AFL (SelectKBest + imbalance strategies)."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / os.environ.get("AF_AFL_OUT", "af_afl_full")
FEAT = os.environ.get("AF_AFL_FEAT", "v1")
SUFFIX = "" if (DATA.name == "af_afl_full" and FEAT == "v1") else f"_{DATA.name}_{FEAT}"
FIG = ROOT / "reports" / "figures" / f"af_afl_selected{SUFFIX}"
MET = ROOT / "reports" / "metrics" / f"af_afl_selected{SUFFIX}"
for d in (FIG, MET):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

STATE = 42
KS = [50, 100, 200, 400]
THRESHOLDS = np.linspace(0.15, 0.85, 71)


def smote(X, y, rng, k=5):
    maj = int((y == 0).sum()); mn = int((y == 1).sum())
    if mn < 2 or maj <= mn:
        return X, y
    Xm = X[y == 1]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    inds = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    need = maj - mn
    idx = rng.integers(0, len(Xm), need)
    synth = np.array([Xm[i] + rng.random() * (Xm[rng.choice(inds[i])] - Xm[i]) for i in idx])
    return np.concatenate([X, synth]), np.concatenate([y, np.ones(need, dtype=int)])


def main() -> int:
    ff = "features_advanced.npy" if FEAT == "v1" else f"features_advanced_{FEAT}.npy"
    X = np.load(DATA / ff)
    y = np.load(DATA / "y.npy").astype(int)
    sp = pd.read_csv(DATA / "split.csv")
    tr, va, te = [sp.loc[sp.split == s, "idx"].to_numpy() for s in ("train", "val", "test")]
    an = np.isnan(X[tr]).all(axis=0)
    X = np.delete(X, np.where(an)[0], axis=1)
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    m2 = np.median(X[tr], axis=0)
    X = np.clip((X - m2) / sc, -20, 20).astype(np.float32)
    Xtr, Xva, Xte, ytr, yva, yte = X[tr], X[va], X[te], y[tr], y[va], y[te]

    results = []
    for k in KS:
        sel = SelectKBest(f_classif, k=min(k, Xtr.shape[1])).fit(Xtr, ytr)
        A, B, C = sel.transform(Xtr), sel.transform(Xva), sel.transform(Xte)
        print(f"\n=== k={k} ({A.shape[1]} features) ===", flush=True)
        for strategy in ("base", "class_weight", "oversample"):
            rng = np.random.default_rng(STATE)
            Xs, ys = A, ytr
            if strategy == "oversample":
                need = int((ys == 0).sum()) - int((ys == 1).sum())
                idx = rng.integers(0, int((ys == 1).sum()), need)
                Xs = np.concatenate([A, A[ys == 1][idx]])
                ys = np.concatenate([ys, np.ones(need, dtype=int)])
            cw = "balanced" if strategy == "class_weight" else None
            models = {
                "LogReg": LogisticRegression(max_iter=1500, class_weight=cw, n_jobs=-1, random_state=STATE),
                "SVC": SVC(kernel="rbf", C=2.0, probability=True, class_weight=cw, random_state=STATE),
                "HistGB": HistGradientBoostingClassifier(max_iter=200, class_weight=cw, random_state=STATE),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight=cw,
                                                   n_jobs=-1, random_state=STATE),
            }
            for name, model in models.items():
                t0 = time.time()
                model.fit(Xs, ys)
                pv = model.predict_proba(B)[:, 1]
                pt = model.predict_proba(C)[:, 1]
                thr = float(THRESHOLDS[int(np.argmax([f1_score(yva, (pv >= t).astype(int), average="macro")
                                                      for t in THRESHOLDS]))])
                yp = (pt >= thr).astype(int)
                r = {"k": k, "strategy": strategy, "model": name, "threshold": thr,
                     "f1_macro": float(f1_score(yte, yp, average="macro")),
                     "f1_af": float(f1_score(yte, yp, pos_label=1)),
                     "accuracy": float(accuracy_score(yte, yp)),
                     "auc_roc": float(roc_auc_score(yte, pt)),
                     "auc_pr": float(average_precision_score(yte, pt)),
                     "mcc": float(matthews_corrcoef(yte, yp)),
                     "time": time.time() - t0}
                results.append(r)
                print(f"  {strategy:12s} {name:11s}: F1={r['f1_macro']:.4f} AUC={r['auc_roc']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(MET / "selected_results.csv", index=False)
    pv = df.pivot_table(index="k", columns="model", values="auc_roc")
    pv.plot(kind="bar", figsize=(11, 5), title="AUC by number of selected features")
    plt.tight_layout()
    plt.savefig(FIG / "auc_by_k.png", dpi=150)
    plt.close()
    (MET / "selected_report.txt").write_text(df.sort_values("f1_macro", ascending=False).to_string(index=False),
                                              encoding="utf-8")
    print("\nBEST:", df.sort_values("f1_macro", ascending=False).head(5).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
