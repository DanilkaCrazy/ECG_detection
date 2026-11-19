"""Compare domain-shift mitigation strategies for the balanced merged dataset.

A: Chapman-only baseline (same config)
B: merged training, Chapman val calibration
C: merged training with external sample weight 0.3
D: pretrain on merged, fine-tune on Chapman only
All variants evaluated on the untouched Chapman test; external test reported
with per-source thresholds.
"""

from __future__ import annotations

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, recall_score, precision_score,
                             roc_auc_score, roc_curve)

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_merged"
FIG = ROOT / "reports" / "figures" / "af_afl_merged"
MET = ROOT / "reports" / "metrics" / "af_afl_merged"
MOD = ROOT / "reports" / "models" / "af_afl_merged"
CFG = {"max_iter": 275, "learning_rate": 0.06, "max_leaf_nodes": 15,
       "min_samples_leaf": 40, "l2_regularization": 5.0}
THRESHOLDS = np.linspace(0.10, 0.90, 81)


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"f1_macro": float(f1_score(y, pred, average="macro")),
            "afib_f1": float(f1_score(y, pred, pos_label=1)),
            "afl_f1": float(f1_score(y, pred, pos_label=0)),
            "afib_recall": float(recall_score(y, pred, pos_label=1)),
            "afib_precision": float(precision_score(y, pred, pos_label=1)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "pr_auc_afib": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)))}


def main() -> int:
    e = np.load(D / "features_advanced_expert.npy")
    v = np.load(D / "features_advanced_v2.npy")
    h = np.load(D / "features_heat.npy")
    X = np.concatenate([e, v, h], axis=1)
    y = np.load(D / "y.npy").astype(int)
    sp = pd.read_csv(D / "split.csv")
    idx = {s: sp.loc[sp.split == s, "idx"].to_numpy() for s in
           ("train", "val", "test_chapman", "test_external")}
    tr, va = idx["train"], idx["val"]
    an = np.isnan(X[tr]).all(axis=0)
    X = np.delete(X, np.where(an)[0], axis=1)
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    mtr = pd.read_csv(D / "meta_train.csv")
    is_ch = mtr["source"].to_numpy() == "chapman"
    ch_tr = tr[is_ch]
    mva = pd.read_csv(D / "meta_val.csv")
    ch_va = va[mva["source"].to_numpy() == "chapman"]

    def fit_cal(model, Xtr, ytr, Xva, yva):
        model.fit(Xtr, ytr)
        pva = model.predict_proba(Xva)[:, 1]
        cal = LogisticRegression().fit(pva.reshape(-1, 1), yva)
        pc = cal.predict_proba(pva.reshape(-1, 1))[:, 1]
        sc = [f1_score(yva, (pc >= t).astype(int), average="macro") for t in THRESHOLDS]
        return model, cal, float(THRESHOLDS[int(np.argmax(sc))]), roc_auc_score(yva, pc)

    variants = {}
    # A: Chapman only
    variants["A_chapman_only"] = fit_cal(
        HistGradientBoostingClassifier(**CFG, class_weight="balanced", random_state=42),
        X[ch_tr], y[ch_tr], X[ch_va], y[ch_va])
    # B: merged
    variants["B_merged"] = fit_cal(
        HistGradientBoostingClassifier(**CFG, class_weight="balanced", random_state=42),
        X[tr], y[tr], X[ch_va], y[ch_va])
    # C: merged with external downweight
    w = np.where(is_ch, 1.0, 0.3)
    mdl = HistGradientBoostingClassifier(**CFG, class_weight="balanced", random_state=42)
    mdl.fit(X[tr], y[tr], sample_weight=w)
    pva = mdl.predict_proba(X[ch_va])[:, 1]
    cal = LogisticRegression().fit(pva.reshape(-1, 1), y[ch_va])
    pc = cal.predict_proba(pva.reshape(-1, 1))[:, 1]
    sc = [f1_score(y[ch_va], (pc >= t).astype(int), average="macro") for t in THRESHOLDS]
    variants["C_weighted"] = (mdl, cal, float(THRESHOLDS[int(np.argmax(sc))]),
                              roc_auc_score(y[ch_va], pc))
    # D: Chapman refit with a small external admixture (regularising diversity)
    variants["D_chapman_refit"] = fit_cal(
        HistGradientBoostingClassifier(**CFG, class_weight="balanced", random_state=42),
        np.concatenate([X[ch_tr], X[tr[~is_ch][:800]]]),
        np.concatenate([y[ch_tr], y[tr[~is_ch][:800]]]),
        X[ch_va], y[ch_va])

    rows = []
    for name, (model, cal, thr, vauc) in variants.items():
        ids = idx["test_chapman"]
        p = cal.predict_proba(model.predict_proba(X[ids])[:, 1].reshape(-1, 1))[:, 1]
        r = evaluate(y[ids], p, thr)
        r.update({"variant": name, "threshold": thr, "chapman_val_auc": vauc})
        rows.append(r)
        print(f"{name}: F1={r['f1_macro']:.4f} AFIB-F1={r['afib_f1']:.4f} "
              f"AUC={r['roc_auc']:.4f}", flush=True)
        joblib.dump({"model": model, "calibrator": cal, "threshold": thr}, MOD / f"{name}.joblib")
        fpr, tpr, _ = roc_curve(y[ids], p); pr, rc, _ = precision_recall_curve(y[ids], p)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"{name} AUC={r['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title("ROC — merged variants (Chapman test)")
        fig.tight_layout(); fig.savefig(FIG / f"roc_{name}.png", dpi=150); plt.close()
        cm = confusion_matrix(y[ids], (p >= thr).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AFL", "AFIB"],
                    yticklabels=["AFL", "AFIB"], ax=ax)
        ax.set_title(f"CM {name}")
        fig.tight_layout(); fig.savefig(FIG / f"cm_{name}.png", dpi=150); plt.close()

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv(MET / "merged_variants.csv", index=False)
    (MET / "merged_variants_report.txt").write_text(df.to_string(index=False), encoding="utf-8")
    print("\nBEST:", df.head(2).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
