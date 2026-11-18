"""Compare imbalance-handling strategies for AF vs AFL classification.

Uses the FULL AF/AFL dataset (all valid records, ~9.6k) and advanced features.
Strategies: none, class_weight, random oversampling, random undersampling,
SMOTE, SMOTE+class_weight, and decision-threshold tuning.
Models: LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting,
KNN, SVC(RBF).

All results, figures (class histograms, ROC, PR, confusion matrices, metric
comparisons, feature importance, threshold curves) go to reports/.
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
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / os.environ.get("AF_AFL_OUT", "af_afl_full")
FEAT_VERSION = os.environ.get("AF_AFL_FEAT", "v1")
SUFFIX = "" if (FEAT_VERSION == "v1" and DATA.name == "af_afl_full") else f"_{DATA.name}_{FEAT_VERSION}"
FIG = ROOT / "reports" / "figures" / f"af_afl_imbalance{SUFFIX}"
MET = ROOT / "reports" / "metrics" / f"af_afl_imbalance{SUFFIX}"
MOD = ROOT / "reports" / "models" / f"af_afl_imbalance{SUFFIX}"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
STRATEGIES = ["base", "class_weight", "oversample", "undersample", "smote", "smote_cw"]
THRESHOLDS = np.linspace(0.15, 0.85, 71)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, rng: np.random.Generator | None = None):
    """Minimal SMOTE: interpolate each synthetic point with a random minority neighbour."""
    rng = rng or np.random.default_rng(RANDOM_STATE)
    maj_n = int((y == 0).sum())
    min_n = int((y == 1).sum())
    if min_n < 2 or maj_n <= min_n:
        return X, y
    Xm = X[y == 1]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    inds = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    need = maj_n - min_n
    synth = np.empty((need, X.shape[1]), dtype=X.dtype)
    base_idx = rng.integers(0, len(Xm), size=need)
    for t, bi in enumerate(base_idx):
        nj = rng.choice(inds[bi])
        gap = rng.random()
        synth[t] = Xm[bi] + gap * (Xm[nj] - Xm[bi])
    return np.concatenate([X, synth], axis=0), np.concatenate([y, np.ones(need, dtype=y.dtype)])


def resample(X: np.ndarray, y: np.ndarray, strategy: str, rng):
    if strategy in ("base", "class_weight"):
        return X, y
    if strategy == "oversample":
        Xm, ym = X[y == 1], y[y == 1]
        need = int((y == 0).sum()) - len(ym)
        idx = rng.integers(0, len(Xm), size=max(need, 0))
        return np.concatenate([X, Xm[idx]]), np.concatenate([y, ym[idx]])
    if strategy == "undersample":
        X0, y0 = X[y == 0], y[y == 0]
        idx = rng.choice(len(X0), size=int((y == 1).sum()), replace=False)
        return np.concatenate([X[y == 1], X0[idx]]), np.concatenate([y[y == 1], y0[idx]])
    if strategy in ("smote", "smote_cw"):
        return smote(X, y, rng=rng)
    raise ValueError(strategy)


def make_models(strategy: str):
    cw = "balanced" if strategy in ("class_weight", "smote_cw") else None
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE),
        "RandForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight=cw,
                                             n_jobs=-1, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight=cw,
                                           n_jobs=-1, random_state=RANDOM_STATE),
        "HistGB": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, class_weight=cw,
                                                 random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        "SVC_RBF": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight=cw,
                       random_state=RANDOM_STATE),
    }


def main() -> int:
    feat_file = "features_advanced.npy" if FEAT_VERSION == "v1" else f"features_advanced_{FEAT_VERSION}.npy"
    X = np.load(DATA / feat_file)
    y = np.load(DATA / "y.npy").astype(int)
    split = pd.read_csv(DATA / "split.csv")
    tr = split.loc[split["split"] == "train", "idx"].to_numpy()
    va = split.loc[split["split"] == "val", "idx"].to_numpy()
    te = split.loc[split["split"] == "test", "idx"].to_numpy()
    print(f"Feature matrix: {X.shape}, train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    # clean: drop all-NaN, fill rest with train median
    allnan = np.isnan(X[tr]).all(axis=0)
    if allnan.any():
        X = np.delete(X, np.where(allnan)[0], axis=1)
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)
    # robust z-score fit on train
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    scale = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    med2 = np.median(X[tr], axis=0)
    X = np.clip((X - med2) / scale, -20, 20).astype(np.float32)
    joblib.dump({"allnan_cols": np.where(allnan)[0], "median_fill": med,
                 "center": med2, "scale": scale}, MOD / "preprocessing.joblib")
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    print(f"After cleaning: {X.shape}; AF train={int(ytr.sum())}, AFL train={int((ytr==0).sum())}", flush=True)

    # ---------- figure: class distribution for every strategy ----------
    rng = np.random.default_rng(RANDOM_STATE)
    distributions: dict[str, tuple[int, int]] = {}
    for s in STRATEGIES:
        _, ys = resample(Xtr, ytr, s, rng)
        distributions[s] = (int((ys == 0).sum()), int((ys == 1).sum()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, s in zip(axes.ravel(), STRATEGIES):
        a, b = distributions[s]
        ax.bar(["AFL", "AF"], [a, b], color=["#3498db", "#e74c3c"])
        ax.set_title(f"{s}: AFL={a}, AF={b}")
        ax.set_ylim(0, max(a, b) * 1.15)
    fig.suptitle("Class balance before/after imbalance strategies (training set)")
    savefig("class_distribution_strategies.png")

    results: list[dict] = []
    best_overall = None
    for s in STRATEGIES:
        Xs, ys = resample(Xtr, ytr, s, rng)
        print(f"\n=== Strategy: {s} (n={len(ys)}, AF={int(ys.sum())}) ===", flush=True)
        fig_r, ax_r = plt.subplots(figsize=(7, 6))
        fig_p, ax_p = plt.subplots(figsize=(7, 6))
        for mname, model in make_models(s).items():
            t0 = time.time()
            model.fit(Xs, ys)
            pv = model.predict_proba(Xva)[:, 1]
            pt_base = model.predict_proba(Xte)[:, 1]
            auc = float(roc_auc_score(yte, pt_base))
            ap = float(average_precision_score(yte, pt_base))
            f1_base = float(f1_score(yte, (pt_base >= 0.5).astype(int), average="macro"))
            # threshold tuning on validation (max macro F1)
            scores = [f1_score(yva, (pv >= t).astype(int), average="macro") for t in THRESHOLDS]
            thr = float(THRESHOLDS[int(np.argmax(scores))])
            yp = (pt_base >= thr).astype(int)
            res = {
                "strategy": s, "model": mname,
                "threshold": thr,
                "accuracy": float(accuracy_score(yte, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yte, yp)),
                "f1_macro": float(f1_score(yte, yp, average="macro")),
                "f1_af": float(f1_score(yte, yp, pos_label=1)),
                "f1_afl": float(f1_score(yte, yp, pos_label=0)),
                "precision_af": float(precision_score(yte, yp, pos_label=1)),
                "recall_af": float(recall_score(yte, yp, pos_label=1)),
                "mcc": float(matthews_corrcoef(yte, yp)),
                "kappa": float(cohen_kappa_score(yte, yp)),
                "log_loss": float(log_loss(yte, np.clip(pt_base, 1e-12, 1 - 1e-12))),
                "auc_roc": auc, "auc_pr": ap,
                "f1_macro_at_0.5": f1_base,
                "train_time": time.time() - t0,
            }
            results.append(res)
            fpr, tpr, _ = roc_curve(yte, pt_base)
            ax_r.plot(fpr, tpr, label=f"{mname} (AUC={auc:.3f})")
            pr, rc, _ = precision_recall_curve(yte, pt_base)
            ax_p.plot(rc, pr, label=f"{mname} (AP={ap:.3f})")
            print(f"  {mname:11s}: F1={res['f1_macro']:.4f} AF-F1={res['f1_af']:.4f} "
                  f"AUC={auc:.4f} AP={ap:.4f} thr={thr:.2f} ({res['train_time']:.1f}s)", flush=True)
            if best_overall is None or res["f1_macro"] > best_overall[0]["f1_macro"]:
                best_overall = (res, model, s, mname, thr, pt_base, yte)
        ax_r.plot([0, 1], [0, 1], "k--")
        ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
        ax_r.set_title(f"ROC — {s}"); ax_r.legend(fontsize=8)
        fig_r.tight_layout(); fig_r.savefig(FIG / f"roc_{s}.png", dpi=150); plt.close(fig_r)
        ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
        ax_p.set_title(f"PR — {s}"); ax_p.legend(fontsize=8)
        fig_p.tight_layout(); fig_p.savefig(FIG / f"pr_{s}.png", dpi=150); plt.close(fig_p)

    df = pd.DataFrame(results)
    df.to_csv(MET / "imbalance_results.csv", index=False)

    # ---------- metric comparison figures ----------
    for metric in ("f1_macro", "f1_af", "auc_roc", "auc_pr", "balanced_accuracy"):
        pivot = df.pivot(index="model", columns="strategy", values=metric)
        plt.figure(figsize=(12, 6))
        pivot.plot(kind="bar")
        plt.title(f"{metric} by imbalance strategy")
        plt.ylabel(metric); plt.xticks(rotation=30, ha="right")
        plt.legend(title="strategy", fontsize=8)
        savefig(f"comparison_{metric}.png")

    # ---------- best model artifacts ----------
    res, model, s, mname, thr, prob, ytrue = best_overall
    ypred = (prob >= thr).astype(int)
    cm = confusion_matrix(ytrue, ypred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AFL", "AF"], yticklabels=["AFL", "AF"])
    plt.title(f"Best: {mname} / {s} (F1={res['f1_macro']:.4f})")
    plt.ylabel("True"); plt.xlabel("Predicted")
    savefig("cm_best.png")
    joblib.dump({"model": model, "threshold": thr, "strategy": s, "model_name": mname,
                 "features": "advanced", "classes": ["AFL", "AF"]}, MOD / "best_model.joblib")

    # feature importance if tree-based
    if hasattr(model, "feature_importances_"):
        names = pd.read_csv(DATA / "feature_names.csv")["name"].tolist()
        names = [n for i, n in enumerate(names) if i < len(model.feature_importances_)]
        imp = model.feature_importances_[: len(names)]
        top = np.argsort(imp)[-25:][::-1]
        plt.figure(figsize=(10, 8))
        sns.barplot(x=imp[top], y=[names[i] for i in top], palette="viridis")
        plt.title("Top-25 feature importances (best tree model)")
        savefig("feature_importance.png")

    # threshold curve for best
    f1s = [f1_score(ytrue, (prob >= t).astype(int), average="macro") for t in THRESHOLDS]
    plt.figure(figsize=(8, 5))
    plt.plot(THRESHOLDS, f1s)
    plt.axvline(thr, color="r", ls="--", label=f"chosen={thr:.2f}")
    plt.xlabel("Decision threshold"); plt.ylabel("Macro F1 (test)")
    plt.title(f"Threshold sensitivity — {mname}/{s}")
    plt.legend()
    savefig("threshold_curve.png")

    # ---------- report ----------
    af_n, afl_n = int(y.sum()), int((y == 0).sum())
    lines = [
        "AF vs AFL — FULL DATA IMBALANCE STUDY",
        "=" * 70,
        f"Records: {len(y)} (AF={af_n}, AFL={afl_n}, ratio={afl_n/af_n:.2f})",
        f"Features: {X.shape[1]} (base + spectral + RR/peaks + morphology + autocorrelation)",
        f"Split: train={len(tr)}, val={len(va)}, test={len(te)} (group-aware)",
        f"Imbalance strategies: {STRATEGIES}",
        "Threshold tuned on validation (max macro F1), reported on test.",
        "",
        "CLASS DISTRIBUTION AFTER EACH STRATEGY (train):",
    ]
    for s, (a, b) in distributions.items():
        lines.append(f"  {s:14s}: AFL={a:5d}, AF={b:5d}, ratio={a/max(b,1):.2f}")
    lines += ["", "RESULTS (test set):", df.sort_values("f1_macro", ascending=False).to_string(index=False)]
    lines += ["", f"BEST: {mname} / {s}, threshold={thr:.2f}, F1_macro={res['f1_macro']:.4f}, "
                   f"AF-F1={res['f1_af']:.4f}, AUC={res['auc_roc']:.4f}, AP={res['auc_pr']:.4f}"]
    (MET / "imbalance_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nBEST:", res, flush=True)
    print("Done!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
