"""Train the final HistGB baseline on the balanced merged dataset.

Evaluation on two untouched tests:
- test_chapman: in-domain independent Chapman strict test
- test_external: out-of-domain PTB-XL/CPSC2018 holdout
Compared to the Stage 2 Chapman-only baseline (Macro-F1 0.609, AFIB-F1 0.443).
"""

from __future__ import annotations

import json
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
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_merged"
FIG = ROOT / "reports" / "figures" / "af_afl_merged"
MET = ROOT / "reports" / "metrics" / "af_afl_merged"
MOD = ROOT / "reports" / "models" / "af_afl_merged"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

THRESHOLDS = np.linspace(0.10, 0.90, 81)
CFG = {"max_iter": 275, "learning_rate": 0.06, "max_leaf_nodes": 15,
       "min_samples_leaf": 40, "l2_regularization": 5.0}


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"f1_macro": f1_score(y, pred, average="macro"),
            "afib_f1": f1_score(y, pred, pos_label=1),
            "afl_f1": f1_score(y, pred, pos_label=0),
            "afib_recall": recall_score(y, pred, pos_label=1),
            "afib_precision": precision_score(y, pred, pos_label=1),
            "afl_recall": recall_score(y, pred, pos_label=0),
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "mcc": matthews_corrcoef(y, pred),
            "pr_auc_afib": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)),
            "brier": brier_score_loss(y, p)}


def main() -> int:
    e = np.load(D / "features_advanced_expert.npy")
    v = np.load(D / "features_advanced_v2.npy")
    h = np.load(D / "features_heat.npy")
    X = np.concatenate([e, v, h], axis=1)
    en = pd.read_csv(D / "feature_names_expert.csv")["name"].tolist()
    vn = pd.read_csv(D / "feature_names_v2.csv")["name"].tolist()
    hn = pd.read_csv(D / "feature_names_heat.csv")["name"].tolist()
    names = [f"expert_{n}" for n in en] + [f"v2_{n}" for n in vn] + [f"heat_{n}" for n in hn]
    y = np.load(D / "y.npy").astype(int)
    sp = pd.read_csv(D / "split.csv")
    idx = {s: sp.loc[sp.split == s, "idx"].to_numpy() for s in
           ("train", "val", "test_chapman", "test_external")}
    tr, va = idx["train"], idx["val"]
    an = np.isnan(X[tr]).all(axis=0)
    X = np.delete(X, np.where(an)[0], axis=1)
    names = [n for i, n in enumerate(names) if i not in set(np.where(an)[0].tolist())]
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]
    print(f"Merged: train={len(tr)} (AFIB {int(ytr.sum())}), val={len(va)}, features={X.shape[1]}", flush=True)

    model = HistGradientBoostingClassifier(**CFG, class_weight="balanced",
                                           early_stopping=True, validation_fraction=0.15,
                                           random_state=42)
    model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    cal = LogisticRegression().fit(pva.reshape(-1, 1), yva)
    pva_c = cal.predict_proba(pva.reshape(-1, 1))[:, 1]
    scores = [f1_score(yva, (pva_c >= t).astype(int), average="macro") for t in THRESHOLDS]
    thr = float(THRESHOLDS[int(np.argmax(scores))])
    print(f"val AUC={roc_auc_score(yva, pva_c):.4f}, global threshold={thr:.3f}", flush=True)

    # source-specific calibration + threshold (source is known at inference)
    mva = pd.read_csv(D / "meta_val.csv")
    src_va = mva["source"].to_numpy()
    cal_src, thr_src = {}, {}
    for src in np.unique(src_va):
        m = src_va == src
        key = "chapman" if "chapman" in str(src) else "external"
        if len(np.unique(yva[m])) < 2:
            cal_src[key] = cal
            thr_src[key] = 0.5
            print(f"  source {src}: n={m.sum()}, single class -> global calibration, thr=0.5", flush=True)
            continue
        c = LogisticRegression().fit(pva[m].reshape(-1, 1), yva[m])
        pc = c.predict_proba(pva[m].reshape(-1, 1))[:, 1]
        sc = [f1_score(yva[m], (pc >= t).astype(int), average="macro") for t in THRESHOLDS]
        cal_src[key] = c
        thr_src[key] = float(THRESHOLDS[int(np.argmax(sc))])
        print(f"  source {src}: n={m.sum()}, val AUC={roc_auc_score(yva[m], pc):.4f}, "
              f"threshold={thr_src[key]:.3f}", flush=True)

    rows = []
    for name in ("test_chapman", "test_external"):
        ids = idx[name]
        dom = "chapman" if name == "test_chapman" else "external"
        p = cal_src[dom].predict_proba(model.predict_proba(X[ids])[:, 1].reshape(-1, 1))[:, 1]
        thr = thr_src[dom]
        r = evaluate(y[ids], p, thr)
        r["test"] = name; r["threshold"] = thr
        rows.append(r)
        print(name, {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                     for k, v in r.items()}, flush=True)
        meta = pd.read_csv(D / f"meta_{name}.csv")
        out = meta[["record_id", "source", "rhythm"]].copy()
        out["y_true"] = y[ids]; out["prob_afib"] = p; out["pred"] = (p >= thr).astype(int)
        out.to_csv(MET / f"predictions_{name}.csv", index=False)
        fpr, tpr, _ = roc_curve(y[ids], p); pr, rc, _ = precision_recall_curve(y[ids], p)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={r['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"ROC — {name}")
        fig.tight_layout(); fig.savefig(FIG / f"roc_{name}.png", dpi=150); plt.close()
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(rc, pr, label=f"AP={r['pr_auc_afib']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title(f"PR — {name}")
        fig.tight_layout(); fig.savefig(FIG / f"pr_{name}.png", dpi=150); plt.close()
        cm = confusion_matrix(y[ids], (p >= thr).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AFL", "AFIB"],
                    yticklabels=["AFL", "AFIB"], ax=ax)
        ax.set_title(f"CM — {name} F1={r['f1_macro']:.3f}")
        fig.tight_layout(); fig.savefig(FIG / f"cm_{name}.png", dpi=150); plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(MET / "merged_results.csv", index=False)
    joblib.dump({"model": model, "calibrator": cal, "threshold": thr, "cfg": CFG,
                 "features": "combined_heat691", "names": names}, MOD / "merged_model.joblib")
    lines = [
        "# Balanced merged dataset — AFIB/AFL", "",
        f"Records: {len(y)}; AFIB {int((y==1).sum())}, AFL {int((y==0).sum())}, "
        f"ratio {((y==0).sum()/(y==1).sum()):.3f}",
        f"Train: {len(tr)} (AFIB {int(ytr.sum())}), Val: {len(va)}",
        f"Sources: chapman 7373, ptbxl 1553, cpsc2018 1221, cpsc2018_extra 207",
        f"Threshold (val): {thr:.3f}", "",
        "Stage 2 Chapman-only baseline: Macro-F1 0.6094, AFIB-F1 0.4413, AUC 0.6785", "",
        df.to_string(index=False),
    ]
    (MET / "merged_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
