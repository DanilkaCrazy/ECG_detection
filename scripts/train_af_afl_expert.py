"""Baseline on physiologically-motivated expert features (dedup, leak-free data).

Compares three feature sets: expert-only (AV conduction, entropy, spectral),
v2-only, and combined. Models: LogReg, HistGB, SVC, ExtraTrees, RF.
Imbalance: base / class_weight / oversample. Threshold tuned on validation.
All reported metrics (including AUC) are on the test set.
"""

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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.svm import SVC

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / "af_afl_full10s_dedup"
FIG = ROOT / "reports" / "figures" / "af_afl_expert"
MET = ROOT / "reports" / "metrics" / "af_afl_expert"
MOD = ROOT / "reports" / "models" / "af_afl_expert"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

STATE = 42
THRESHOLDS = np.linspace(0.15, 0.85, 71)
CLASSES = ["AFL", "AF"]  # 0=AFL, 1=AF (AF is the minority here)


def load_set(kind: str):
    if kind == "expert":
        X = np.load(DATA / "features_advanced_expert.npy")
        names = pd.read_csv(DATA / "feature_names_expert.csv")["name"].tolist()
    elif kind == "v2":
        X = np.load(DATA / "features_advanced_v2.npy")
        names = pd.read_csv(DATA / "feature_names_v2.csv")["name"].tolist()
    else:
        e = np.load(DATA / "features_advanced_expert.npy")
        v = np.load(DATA / "features_advanced_v2.npy")
        en = pd.read_csv(DATA / "feature_names_expert.csv")["name"].tolist()
        vn = pd.read_csv(DATA / "feature_names_v2.csv")["name"].tolist()
        X = np.concatenate([e, v], axis=1)
        names = [f"expert_{n}" for n in en] + [f"v2_{n}" for n in vn]
    return X, names


def main() -> int:
    y = np.load(DATA / "y.npy").astype(int)
    sp = pd.read_csv(DATA / "split.csv")
    tr, va, te = [sp.loc[sp.split == s, "idx"].to_numpy() for s in ("train", "val", "test")]
    print(f"Data: {len(y)} records, AF={int(y.sum())}, AFL={int((y==0).sum())}, "
          f"split {len(tr)}/{len(va)}/{len(te)}", flush=True)

    results = []
    best = None
    for kind in ("expert", "v2", "combined"):
        Xraw, names = load_set(kind)
        an = np.isnan(Xraw[tr]).all(axis=0)
        X = np.delete(Xraw, np.where(an)[0], axis=1)
        keep_names = [n for i, n in enumerate(names) if i not in set(np.where(an)[0].tolist())]
        med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
        X = np.where(np.isnan(X), med, X)
        q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
        scale = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
        center = np.median(X[tr], axis=0)
        X = np.clip((X - center) / scale, -20, 20).astype(np.float32)
        Xtr, Xva, Xte = X[tr], X[va], X[te]
        ytr, yva, yte = y[tr], y[va], y[te]
        print(f"\n=== {kind}: {X.shape[1]} features ===", flush=True)

        for strategy in ("base", "class_weight", "oversample"):
            rng = np.random.default_rng(STATE)
            Xs, ys = Xtr, ytr
            if strategy == "oversample":
                need = int((ytr == 0).sum()) - int((ytr == 1).sum())
                idx = rng.integers(0, int((ytr == 1).sum()), need)
                Xs = np.concatenate([Xtr, Xtr[ytr == 1][idx]])
                ys = np.concatenate([ytr, np.ones(need, dtype=int)])
            cw = "balanced" if strategy == "class_weight" else None
            models = {
                "LogReg": LogisticRegression(max_iter=2000, class_weight=cw, n_jobs=-1, random_state=STATE),
                "HistGB": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                          class_weight=cw, random_state=STATE),
                "SVC": SVC(kernel="rbf", C=2.0, probability=True, class_weight=cw, random_state=STATE),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2,
                                                   class_weight=cw, n_jobs=-1, random_state=STATE),
                "RF": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                             class_weight=cw, n_jobs=-1, random_state=STATE),
            }
            for name, model in models.items():
                t0 = time.time()
                model.fit(Xs, ys)
                pv = model.predict_proba(Xva)[:, 1]
                pt = model.predict_proba(Xte)[:, 1]
                scores = [f1_score(yva, (pv >= t).astype(int), average="macro") for t in THRESHOLDS]
                thr = float(THRESHOLDS[int(np.argmax(scores))])
                yp = (pt >= thr).astype(int)
                r = {
                    "features": kind, "strategy": strategy, "model": name, "threshold": thr,
                    "f1_macro": float(f1_score(yte, yp, average="macro")),
                    "f1_af": float(f1_score(yte, yp, pos_label=1)),
                    "f1_afl": float(f1_score(yte, yp, pos_label=0)),
                    "af_recall": float(recall_score(yte, yp, pos_label=1)),
                    "af_precision": float(precision_score(yte, yp, pos_label=1)),
                    "afl_recall": float(recall_score(yte, yp, pos_label=0)),
                    "afl_precision": float(precision_score(yte, yp, pos_label=0)),
                    "accuracy": float(accuracy_score(yte, yp)),
                    "balanced_accuracy": float(balanced_accuracy_score(yte, yp)),
                    "mcc": float(matthews_corrcoef(yte, yp)),
                    "auc_roc": float(roc_auc_score(yte, pt)),
                    "auc_pr": float(average_precision_score(yte, pt)),
                    "log_loss": float(log_loss(yte, np.clip(pt, 1e-12, 1 - 1e-12))),
                    "time": time.time() - t0,
                }
                results.append(r)
                print(f"  {strategy:12s} {name:11s}: F1={r['f1_macro']:.4f} AF={r['f1_af']:.4f} "
                      f"AFL={r['f1_afl']:.4f} AUC={r['auc_roc']:.4f}", flush=True)
                if best is None or r["f1_macro"] > best[0]["f1_macro"]:
                    best = (r, model, kind, strategy, name, thr, pt, yte, keep_names,
                            np.where(an)[0], med, center, scale)

    df = pd.DataFrame(results)
    df.to_csv(MET / "expert_results.csv", index=False)
    for metric in ("f1_macro", "f1_af", "auc_roc", "auc_pr"):
        pv = df.pivot_table(index="model", columns="features", values=metric)
        pv.plot(kind="bar", figsize=(10, 5), title=f"{metric} by feature set")
        plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(FIG / f"compare_{metric}.png", dpi=150); plt.close()

    r, model, kind, strategy, name, thr, pt, ytrue, knames, anidx, med, center, scale = best
    yp = (pt >= thr).astype(int)
    cm = confusion_matrix(ytrue, yp, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Best: {name}/{strategy}/{kind} F1={r['f1_macro']:.4f}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(FIG / "cm_best.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(ytrue, pt); pr, rc, _ = precision_recall_curve(ytrue, pt)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={r['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title("ROC — best expert model")
    fig.tight_layout(); fig.savefig(FIG / "roc_best.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(rc, pr, label=f"AP={r['auc_pr']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title("PR — best expert model")
    fig.tight_layout(); fig.savefig(FIG / "pr_best.png", dpi=150); plt.close()
    joblib.dump({"model": model, "threshold": thr, "features": kind, "strategy": strategy,
                 "model_name": name, "classes": CLASSES}, MOD / "best_model.joblib")
    joblib.dump({"allnan_cols": anidx, "median_fill": med, "center": center, "scale": scale,
                 "kept_names": knames, "dataset": str(DATA)}, MOD / "preprocessing.joblib")
    (MET / "expert_report.txt").write_text(
        "AF/AFL expert-feature study (dedup, leak-free)\n" + "=" * 60 + "\n"
        + f"AF={int(y.sum())} (minority), AFL={int((y==0).sum())}\n\n"
        + df.sort_values("f1_macro", ascending=False).to_string(index=False), encoding="utf-8")
    print("\nBEST:", r, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
