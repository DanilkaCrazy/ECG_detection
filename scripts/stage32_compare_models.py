"""Stage 3.2 — honest model comparison on ONE audited manifest.

Fixes vs stage3_retrain.py:
- uses af_afl_stage31 (built from manifest_final.csv, no SHDB);
- model selection by VALIDATION only, test evaluated once after freeze;
- cluster bootstrap by patient_key (not by record);
- OOF calibration: split validation into calibration and threshold parts;
- records manifest SHA256 for reproducibility;
- saves confusion matrices, calibration curves, PR curves with baseline,
  probability distributions and per-source metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage31"
MET = ROOT / "reports" / "metrics" / "af_afl_stage32"
FIG = ROOT / "reports" / "figures" / "af_afl_stage32"
MOD = ROOT / "reports" / "models" / "af_afl_stage32"
for d in (MET, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.linspace(0.10, 0.90, 81)
MAN_HASH = json.loads((D / "stats.json").read_text(encoding="utf-8"))["manifest_final_sha256"]

CONFIGS = {
    "E1_expert_HistGB": ("expert", dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
                                       min_samples_leaf=25, l2_regularization=0.5)),
    "E2_v2_HistGB": ("v2", dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
                               min_samples_leaf=25, l2_regularization=0.5)),
    "E3_combined_HistGB": ("combined", dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
                                           min_samples_leaf=25, l2_regularization=0.5)),
    "E4_combined_CatBoost": ("combined", dict(iterations=500, depth=6, learning_rate=0.05,
                                             l2_leaf_reg=3, verbose=False, random_seed=42)),
    "E5_combined_XGBoost": ("combined", dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                                            min_child_weight=15, subsample=0.9,
                                            colsample_bytree=0.85, tree_method="hist",
                                            eval_metric="auc", n_jobs=8)),
    "E6_combined_heat_HistGB": ("combined_heat", dict(max_iter=250, learning_rate=0.06,
                                                     max_leaf_nodes=20, min_samples_leaf=25,
                                                     l2_regularization=0.5)),
    "E7_combined_SVC": ("combined", dict(C=2.0, kernel="rbf", random_state=42)),
}


def load_features(kind):
    parts = []
    if kind in ("expert", "combined", "combined_heat"):
        parts.append(np.load(D / "features_advanced_expert.npy"))
    if kind in ("v2", "combined", "combined_heat"):
        parts.append(np.load(D / "features_advanced_v2.npy"))
    if kind == "combined_heat":
        parts.append(np.load(D / "features_heat.npy"))
    return np.concatenate(parts, axis=1)


def build(name, params, ytr):
    pos, neg = float((ytr == 1).sum()), float((ytr == 0).sum())
    if name.endswith("HistGB"):
        return HistGradientBoostingClassifier(**params, class_weight="balanced", random_state=42)
    if name.endswith("CatBoost"):
        return CatBoostClassifier(**params, loss_function="Logloss", eval_metric="AUC",
                                  auto_class_weights="Balanced", thread_count=8)
    if name.endswith("XGBoost"):
        return XGBClassifier(**params, scale_pos_weight=neg / pos, random_state=42)
    if name.endswith("SVC"):
        return SVC(**params, class_weight="balanced", probability=True)
    raise ValueError(name)


def cluster_bootstrap_ci(y, p, thr, groups, key, n=400, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        sel = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([by[g] for g in sel])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = p[idx]
        vals.append(f1_score(yy, (pp >= thr).astype(int), average="macro") if key == "f1_macro"
                    else f1_score(yy, (pp >= thr).astype(int), pos_label=1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    man = pd.read_csv(D / "meta.csv")
    y = man.label.to_numpy()
    groups = man.patient_key.to_numpy()
    src = man.source.to_numpy()
    sp = man.split.to_numpy()
    tr, va, te = np.where(sp == "train")[0], np.where(sp == "val")[0], np.where(sp == "test")[0]
    # split val into calibration and threshold halves by patient
    rng = np.random.default_rng(42)
    vg = np.unique(groups[va])
    cg = rng.choice(vg, len(vg) // 2, replace=False)
    cal_idx = va[np.isin(groups[va], cg)]
    thr_idx = va[~np.isin(groups[va], cg)]
    print(f"train={len(tr)} cal={len(cal_idx)} thr={len(thr_idx)} test={len(te)}", flush=True)

    cache = {}
    rows = []
    only = os.environ.get("ONLY_MODEL")
    for name, (kind, params) in CONFIGS.items():
        if only and name not in only.split(","):
            continue
        if kind not in cache:
            X = load_features(kind)
            an = np.isnan(X[tr]).all(axis=0)
            X = np.delete(X, np.where(an)[0], axis=1)
            med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
            X = np.where(np.isnan(X), med, X)
            q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
            sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
            cen = np.median(X[tr], axis=0)
            X = np.clip((X - cen) / sc, -20, 20).astype(np.float32)
            cache[kind] = X
        X = cache[kind]
        m = build(name, params, y[tr])
        m.fit(X[tr], y[tr])
        pv_cal = m.predict_proba(X[cal_idx])[:, 1]
        pv_thr = m.predict_proba(X[thr_idx])[:, 1]
        # source-specific calibrators fitted on cal, thresholds tuned on separate thr
        cals, thrs = {}, {}
        for s in sorted(set(src[va])):
            mc = src[cal_idx] == s
            mt = src[thr_idx] == s
            if mc.sum() < 10 or len(np.unique(y[cal_idx][mc])) < 2:
                cals[s], thrs[s] = None, 0.5
                continue
            c = LogisticRegression().fit(pv_cal[mc].reshape(-1, 1), y[cal_idx][mc])
            pc = c.predict_proba(pv_thr[mt].reshape(-1, 1))[:, 1] if mt.sum() else pv_thr
            sco = [f1_score(y[thr_idx][mt], (pc >= t).astype(int), average="macro")
                   for t in THRESHOLDS] if mt.sum() else [0] * len(THRESHOLDS)
            cals[s], thrs[s] = c, float(THRESHOLDS[int(np.argmax(sco))])
        # model selection metric from validation (cal+thr predictions with own cal/thr)
        val_pred = np.zeros(len(va), dtype=int)
        for s in sorted(set(src[va])):
            mv = src[va] == s
            pv = m.predict_proba(X[va][mv])[:, 1]
            c = cals.get(s)
            if c is None:
                c = LogisticRegression().fit(pv_cal.reshape(-1, 1), y[cal_idx])
            pc = c.predict_proba(pv.reshape(-1, 1))[:, 1]
            val_pred[mv] = (pc >= thrs.get(s, 0.5)).astype(int)
        val_f1 = f1_score(y[va], val_pred, average="macro")
        print(f"{name}: val F1={val_f1:.4f}", flush=True)

        # TEST — evaluate per source, once
        for s in sorted(set(src[te])):
            mt = src[te] == s
            pv = m.predict_proba(X[te][mt])[:, 1]
            c = cals.get(s)
            if c is None:
                c = LogisticRegression().fit(pv_cal.reshape(-1, 1), y[cal_idx])
            p = c.predict_proba(pv.reshape(-1, 1))[:, 1]
            thr = thrs.get(s, 0.5)
            pred = (p >= thr).astype(int)
            yy = y[te][mt]
            lo, hi = cluster_bootstrap_ci(yy, p, thr, groups[te][mt], "f1_macro")
            r = {
                "model": name, "features": kind, "source": s, "val_f1": val_f1,
                "threshold": thr, "f1_macro": float(f1_score(yy, pred, average="macro")),
                "f1_ci_low": lo, "f1_ci_high": hi,
                "afib_f1": float(f1_score(yy, pred, pos_label=1)),
                "afl_f1": float(f1_score(yy, pred, pos_label=0)),
                "afib_recall": float(recall_score(yy, pred, pos_label=1)),
                "afib_precision": float(precision_score(yy, pred, pos_label=1)),
                "accuracy": float(accuracy_score(yy, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
                "mcc": float(matthews_corrcoef(yy, pred)),
                "pr_auc_afib": float(average_precision_score(yy, p)),
                "pr_baseline": float(yy.mean()),
                "roc_auc": float(roc_auc_score(yy, p)),
                "log_loss": float(log_loss(yy, np.clip(p, 1e-12, 1 - 1e-12))),
                "brier": float(brier_score_loss(yy, p)), "n_test": int(mt.sum()),
                "manifest_sha256": MAN_HASH,
            }
            rows.append(r)
            # figures for the Chapman source only (main in-domain test)
            if s == "chapman":
                cm = confusion_matrix(yy, pred, labels=[0, 1])
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["AFL", "AFIB"], yticklabels=["AFL", "AFIB"])
                plt.title(f"CM {name}"); plt.tight_layout()
                plt.savefig(FIG / f"cm_{name}.png", dpi=150); plt.close()
                fpr, tpr, _ = roc_curve(yy, p); pr, rc, _ = precision_recall_curve(yy, p)
                fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={r['roc_auc']:.3f}")
                ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"ROC {name}")
                fig.tight_layout(); fig.savefig(FIG / f"roc_{name}.png", dpi=150); plt.close()
                fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(rc, pr, label=f"AP={r['pr_auc_afib']:.3f}")
                ax.axhline(float(yy.mean()), ls="--", color="gray", label=f"baseline={yy.mean():.3f}")
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title(f"PR {name}")
                fig.tight_layout(); fig.savefig(FIG / f"pr_{name}.png", dpi=150); plt.close()
                frac, mean = calibration_curve(yy, p, n_bins=8)
                fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(mean, frac, marker="o", label=name)
                ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"Calibration {name}")
                fig.tight_layout(); fig.savefig(FIG / f"cal_{name}.png", dpi=150); plt.close()
        joblib.dump({"model": m, "cals": cals, "thrs": thrs, "kind": kind}, MOD / f"{name}.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage32_results.csv", index=False)
    # select winner by VALIDATION only
    best_val = df.sort_values("val_f1", ascending=False).iloc[0]
    lines = ["# Stage 3.2 — честное сравнение моделей", "",
             f"Манифест SHA256: `{MAN_HASH}`", "",
             "Выбор модели — только по validation F1; тест оценён один раз после фиксации.", "",
             df.to_string(index=False), "",
             f"Лучшая по validation: **{best_val['model']}** (val F1={best_val['val_f1']:.4f}). "
             f"Её Chapman-тест: F1={best_val['f1_macro']:.4f} "
             f"[CI {best_val['f1_ci_low']:.4f}, {best_val['f1_ci_high']:.4f}], "
             f"AFIB-F1={best_val['afib_f1']:.4f}.", ""]
    (ROOT / "reports" / "AF_AFL_STAGE32_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
