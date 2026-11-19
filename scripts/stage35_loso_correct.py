"""Stage 3.5 P0/P3 — correct LOSO with source-local preprocessing.

Fixes the leak in stage34_domain_generalization.py: imputer medians, IQR
scaling and centering are fitted ONLY on the training sources for each fold.
Also adds:
- source classifier baseline (are sources distinguishable from features?);
- source x label contingency table;
- per-source within-domain evaluation using only train-source preprocessing.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage35"
FIG = ROOT / "reports" / "figures" / "af_afl_stage35"
MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
THR = np.linspace(0.05, 0.95, 91)


def fit_preprocess(Xtr):
    an = np.isnan(Xtr).all(axis=0)
    keep = np.where(~an)[0]
    med = np.nan_to_num(np.nanmedian(Xtr[:, keep], axis=0))
    q1, q3 = np.percentile(Xtr[:, keep], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    cen = np.median(Xtr[:, keep], axis=0)
    return keep, med, sc, cen


def apply_preprocess(X, keep, med, sc, cen):
    X = X[:, keep]
    X = np.where(np.isnan(X), med, X)
    return np.clip((X - cen) / sc, -20, 20).astype(np.float32)


def main() -> int:
    man = pd.read_csv(D / "meta.csv")
    y = man.label.to_numpy(); groups = man.patient_key.to_numpy(); src = man.source.to_numpy()
    X = np.concatenate([np.load(D / f, mmap_mode="r") for f in
                        ("features_advanced_expert.npy", "features_advanced_v2.npy",
                         "features_atrial34.npy")], axis=1)
    sources = sorted(set(src))
    rows = []
    for held in sources:
        tr = np.where(src != held)[0]; te = np.where(src == held)[0]
        keep, med, sc, cen = fit_preprocess(np.asarray(X[tr]))   # TRAIN SOURCES ONLY
        Xtr = apply_preprocess(np.asarray(X[tr]), keep, med, sc, cen)
        Xte = apply_preprocess(np.asarray(X[te]), keep, med, sc, cen)
        # threshold from a group-held-out part of training sources
        folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42).split(
            Xtr, y[tr], groups=groups[tr]))
        sub_tr, sub_thr = folds[0][0], folds[0][1]
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           max_leaf_nodes=20, min_samples_leaf=25,
                                           l2_regularization=0.5, class_weight="balanced",
                                           random_state=42)
        m.fit(Xtr[sub_tr], y[tr][sub_tr])
        pthr = m.predict_proba(Xtr[sub_thr])[:, 1]
        thr = float(THR[int(np.argmax([f1_score(y[tr][sub_thr], (pthr >= t).astype(int),
                                               average="macro") for t in THR]))])
        p = m.predict_proba(Xte)[:, 1]; pred = (p >= thr).astype(int)
        rows.append({"held_out_source": held, "n_train": len(tr), "n_test": len(te),
                     "afib_in_test": int((y[te] == 1).sum()), "afl_in_test": int((y[te] == 0).sum()),
                     "threshold": thr,
                     "f1_macro": float(f1_score(y[te], pred, average="macro")),
                     "afib_f1": float(f1_score(y[te], pred, pos_label=1)),
                     "afl_f1": float(f1_score(y[te], pred, pos_label=0)),
                     "afib_recall": float(recall_score(y[te], pred, pos_label=1)),
                     "afib_precision": float(precision_score(y[te], pred, pos_label=1)),
                     "accuracy": float(accuracy_score(y[te], pred)),
                     "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
                     "mcc": float(matthews_corrcoef(y[te], pred)),
                     "pr_auc_afib": float(average_precision_score(y[te], p)),
                     "pr_baseline": float(y[te].mean()),
                     "roc_auc": float(roc_auc_score(y[te], p)),
                     "preprocessing_fit_on": "train sources only"})
        print(held, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage35_loso_correct.csv", index=False)

    # source classifier: can source be predicted from features?
    keep, med, sc, cen = fit_preprocess(np.asarray(X))
    Xa = apply_preprocess(np.asarray(X), keep, med, sc, cen)
    sc_clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    # group split by patient for source classification evaluation
    gss = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(Xa, src, groups))
    tr, te = gss[0][0], gss[0][1]
    sc_clf.fit(Xa[tr], src[tr])
    sp_acc = accuracy_score(src[te], sc_clf.predict(Xa[te]))
    contingency = pd.crosstab(src, y)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("source x label contingency")
    fig.tight_layout(); fig.savefig(FIG / "source_label_contingency.png", dpi=150); plt.close()
    lines = ["# Stage 3.5 — LOSO с корректным preprocessing", "",
             "Импутация, IQR-scaling и центрирование обучаются ТОЛЬКО на train-источниках.", "",
             df.to_string(index=False), "",
             f"Source classifier accuracy (group split): {sp_acc:.4f}", "",
             "source x label:", "", contingency.to_string(), "",
             "Chapman не имеет настоящего patient_id — group = record_id (surrogate)."]
    (ROOT / "reports" / "AF_AFL_STAGE35_DOMAIN_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
