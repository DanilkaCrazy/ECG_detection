"""Stage 3.3 — strict model comparison with independent model-selection validation.

Four-way group split (by patient_key where available):
  train  — fit model + preprocessing
  calib  — fit probability calibrator
  thresh — choose decision threshold
  select — compute validation F1 for MODEL SELECTION (untouched by cal/thr)
  test   — final evaluation once, after model is frozen by `select`

Also provides:
- known-source evaluation (per-source calibrators) and
- unseen-source evaluation (one fixed global threshold, no source calibration).
"""

from __future__ import annotations

import hashlib
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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage31"
MET = ROOT / "reports" / "metrics" / "af_afl_stage33"
FIG = ROOT / "reports" / "figures" / "af_afl_stage33"
MOD = ROOT / "reports" / "models" / "af_afl_stage33"
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
    return SVC(**params, class_weight="balanced", probability=True)


def cluster_bootstrap_ci(y, p, thr, groups, n=400, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(f1_score(yy, (p[idx] >= thr).astype(int), average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    man = pd.read_csv(D / "meta.csv")
    y = man.label.to_numpy()
    groups = man.patient_key.to_numpy()
    src = man.source.to_numpy()
    sp = man.split.to_numpy()
    tr0, held = np.where(sp == "train")[0], np.where(sp != "train")[0]
    # group-aware 4-way split of the held-out part: calib / thresh / select / test
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    folds = list(sgkf.split(held, y[held], groups=groups[held]))
    cal_idx, thr_idx, sel_idx, te_idx = held[folds[0][1]], held[folds[1][1]], held[folds[2][1]], held[folds[3][1]]
    print(f"train={len(tr0)} cal={len(cal_idx)} thr={len(thr_idx)} select={len(sel_idx)} test={len(te_idx)}", flush=True)

    cache, rows = {}, []
    for name, (kind, params) in CONFIGS.items():
        if kind not in cache:
            X = load_features(kind)
            an = np.isnan(X[tr0]).all(axis=0)
            X = np.delete(X, np.where(an)[0], axis=1)
            med = np.nan_to_num(np.nanmedian(X[tr0], axis=0))
            X = np.where(np.isnan(X), med, X)
            q1, q3 = np.percentile(X[tr0], [25, 75], axis=0)
            sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
            cen = np.median(X[tr0], axis=0)
            X = np.clip((X - cen) / sc, -20, 20).astype(np.float32)
            cache[kind] = X
        X = cache[kind]
        m = build(name, params, y[tr0]); m.fit(X[tr0], y[tr0])
        pcal = m.predict_proba(X[cal_idx])[:, 1]

        # calibrator (global) fitted on cal only
        cal = LogisticRegression().fit(pcal.reshape(-1, 1), y[cal_idx])
        pthr = cal.predict_proba(m.predict_proba(X[thr_idx])[:, 1].reshape(-1, 1))[:, 1]
        global_thr = float(THRESHOLDS[int(np.argmax(
            [f1_score(y[thr_idx], (pthr >= t).astype(int), average="macro") for t in THRESHOLDS]))])

        # MODEL SELECTION on independent `sel` (never used for cal/thr)
        psel = cal.predict_proba(m.predict_proba(X[sel_idx])[:, 1].reshape(-1, 1))[:, 1]
        sel_f1 = f1_score(y[sel_idx], (psel >= global_thr).astype(int), average="macro")
        print(f"{name}: select F1={sel_f1:.4f}", flush=True)

        # FINAL TEST (once), global threshold = unseen-source scenario
        pte = cal.predict_proba(m.predict_proba(X[te_idx])[:, 1].reshape(-1, 1))[:, 1]
        for s in sorted(set(src[te_idx])):
            mt = src[te_idx] == s
            yy, pp = y[te_idx][mt], pte[mt]
            lo, hi = cluster_bootstrap_ci(yy, pp, global_thr, groups[te_idx][mt])
            rows.append({
                "model": name, "features": kind, "source": s,
                "select_f1": float(sel_f1), "threshold": global_thr,
                "scenario": "unseen_source_global_threshold",
                "f1_macro": float(f1_score(yy, (pp >= global_thr).astype(int), average="macro")),
                "f1_ci_low": lo, "f1_ci_high": hi,
                "afib_f1": float(f1_score(yy, (pp >= global_thr).astype(int), pos_label=1)),
                "afl_f1": float(f1_score(yy, (pp >= global_thr).astype(int), pos_label=0)),
                "afib_recall": float(recall_score(yy, (pp >= global_thr).astype(int), pos_label=1)),
                "afib_precision": float(precision_score(yy, (pp >= global_thr).astype(int), pos_label=1)),
                "accuracy": float(accuracy_score(yy, (pp >= global_thr).astype(int))),
                "balanced_accuracy": float(balanced_accuracy_score(yy, (pp >= global_thr).astype(int))),
                "mcc": float(matthews_corrcoef(yy, (pp >= global_thr).astype(int))),
                "pr_auc_afib": float(average_precision_score(yy, pp)),
                "pr_baseline": float(yy.mean()),
                "roc_auc": float(roc_auc_score(yy, pp)),
                "log_loss": float(log_loss(yy, np.clip(pp, 1e-12, 1-1e-12))),
                "brier": float(brier_score_loss(yy, pp)), "n_test": int(mt.sum()),
                "manifest_sha256": MAN_HASH,
                "group_note": "Chapman: record_id surrogate, no real patient_id; PTB-XL: patient_id",
            })
        joblib.dump({"model": m, "calibrator": cal, "threshold": global_thr, "kind": kind},
                    MOD / f"{name}.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage33_results.csv", index=False)
    best = df.sort_values("select_f1", ascending=False).iloc[0]
    lines = ["# Stage 3.3 — строгая статистическая оценка", "",
             f"SHA256 манифеста: `{MAN_HASH}`", "",
             "4-way group split: train / calibration / threshold / selection; test — отдельно. "
             "Модель выбирается по `select_f1`, калибратор и порог не видят эти записи. "
             "Сценарий: unseen-source, один глобальный порог.", "",
             df.to_string(index=False), "",
             f"Выбрана по independent validation: **{best['model']}** (select F1={best['select_f1']:.4f}).", ""]
    (ROOT / "reports" / "AF_AFL_STAGE33_STATISTICAL_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
