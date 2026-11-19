"""Stage 3.4 — feature-set experiment with strict protocol and threshold curves.

Feature sets: Expert, V2, Atrial34, Expert+Atrial, V2+Atrial,
Expert+V2, Expert+V2+Atrial, Expert+V2+Atrial+HEAT.
Models: HistGB, CatBoost, XGBoost, LightGBM, SVC, RandomForest.
Same 4-way group split for ALL setups; model chosen on independent selection
split; thresholds: max macro-F1, max AFIB-F1, AFIB recall 0.8/0.9 — all from
threshold split only. Test evaluated once.
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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage34"
FIG = ROOT / "reports" / "figures" / "af_afl_stage34"
MOD = ROOT / "reports" / "models" / "af_afl_stage34"
for p in (MET, FIG, MOD):
    p.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.linspace(0.05, 0.95, 91)
FEATURE_SETS = {
    "expert": ["expert"], "v2": ["v2"], "atrial": ["atrial"],
    "expert_atrial": ["expert", "atrial"], "v2_atrial": ["v2", "atrial"],
    "expert_v2": ["expert", "v2"], "expert_v2_atrial": ["expert", "v2", "atrial"],
    "all_heat": ["expert", "v2", "atrial", "heat"],
}
FILES = {"expert": "features_advanced_expert.npy", "v2": "features_advanced_v2.npy",
         "atrial": "features_atrial34.npy", "heat": "features_heat.npy"}
MODELS = {
    "HistGB": lambda pos, neg: HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=20, min_samples_leaf=25,
        l2_regularization=0.5, class_weight="balanced", random_state=42),
    "CatBoost": lambda pos, neg: CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        verbose=False, thread_count=8, random_seed=42),
    "XGBoost": lambda pos, neg: XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=15,
        subsample=0.9, colsample_bytree=0.85, scale_pos_weight=neg / max(pos, 1),
        tree_method="hist", eval_metric="auc", n_jobs=8, random_state=42),
    "LightGBM": lambda pos, neg: LGBMClassifier(
        n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=25,
        class_weight="balanced", n_jobs=8, verbose=-1, random_state=42),
    "SVC": lambda pos, neg: SVC(C=2.0, kernel="rbf", probability=True,
                               class_weight="balanced", random_state=42),
    "RandomForest": lambda pos, neg: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42),
}


def evaluate(y, p, thr):
    pr = (p >= thr).astype(int)
    return {"f1_macro": float(f1_score(y, pr, average="macro")),
            "afib_f1": float(f1_score(y, pr, pos_label=1)),
            "afl_f1": float(f1_score(y, pr, pos_label=0)),
            "afib_recall": float(recall_score(y, pr, pos_label=1)),
            "afib_precision": float(precision_score(y, pr, pos_label=1)),
            "afl_recall": float(recall_score(y, pr, pos_label=0)),
            "accuracy": float(accuracy_score(y, pr)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pr)),
            "mcc": float(matthews_corrcoef(y, pr)),
            "pr_auc_afib": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1-1e-12))),
            "brier": float(brier_score_loss(y, p))}


def main() -> int:
    man = pd.read_csv(D / "meta.csv")
    y = man.label.to_numpy()
    groups = man.patient_key.to_numpy()
    src = man.source.to_numpy()
    sp = man.split.to_numpy()
    tr0, held = np.where(sp == "train")[0], np.where(sp != "train")[0]
    folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42).split(
        held, y[held], groups=groups[held]))
    cal, thr_i, sel, te = [held[folds[i][1]] for i in range(4)]
    print(f"train={len(tr0)} cal={len(cal)} thr={len(thr_i)} sel={len(sel)} test={len(te)}", flush=True)
    rows, curves = [], {}
    cache = {}
    for fs_name, parts in FEATURE_SETS.items():
        if fs_name not in cache:
            X = np.concatenate([np.load(D / FILES[p], mmap_mode="r") for p in parts], axis=1)
            an = np.isnan(X[tr0]).all(axis=0)
            X = np.delete(X, np.where(an)[0], axis=1)
            med = np.nan_to_num(np.nanmedian(X[tr0], axis=0))
            X = np.where(np.isnan(X), med, X)
            q1, q3 = np.percentile(X[tr0], [25, 75], axis=0)
            sc = np.where((q3-q1) > 1e-9, q3-q1, 1.0)
            cen = np.median(X[tr0], axis=0)
            X = np.clip((X-cen)/sc, -20, 20).astype(np.float32)
            cache[fs_name] = X
        X = cache[fs_name]
        for mname, factory in MODELS.items():
            pos, neg = float((y[tr0] == 1).sum()), float((y[tr0] == 0).sum())
            m = factory(pos, neg)
            m.fit(X[tr0], y[tr0])
            cal_m = LogisticRegression().fit(m.predict_proba(X[cal])[:, 1].reshape(-1, 1), y[cal])
            pthr = cal_m.predict_proba(m.predict_proba(X[thr_i])[:, 1].reshape(-1, 1))[:, 1]
            # threshold variants from threshold split ONLY
            f1m = [f1_score(y[thr_i], (pthr >= t).astype(int), average="macro") for t in THRESHOLDS]
            f1a = [f1_score(y[thr_i], (pthr >= t).astype(int), pos_label=1) for t in THRESHOLDS]
            rec = [recall_score(y[thr_i], (pthr >= t).astype(int), pos_label=1) for t in THRESHOLDS]
            variants = {"max_macro": THRESHOLDS[int(np.argmax(f1m))],
                        "max_afib_f1": THRESHOLDS[int(np.argmax(f1a))],
                        "afib_recall80": THRESHOLDS[int(np.argmin(np.abs(np.array(rec)-0.8)))],
                        "afib_recall90": THRESHOLDS[int(np.argmin(np.abs(np.array(rec)-0.9)))]}
            psel = cal_m.predict_proba(m.predict_proba(X[sel])[:, 1].reshape(-1, 1))[:, 1]
            sel_f1 = f1_score(y[sel], (psel >= variants["max_macro"]).astype(int), average="macro")
            pte = cal_m.predict_proba(m.predict_proba(X[te])[:, 1].reshape(-1, 1))[:, 1]
            for vname, t in variants.items():
                r = evaluate(y[te], pte, t)
                r.update({"features": fs_name, "model": mname, "threshold_variant": vname,
                          "threshold": t, "select_f1": float(sel_f1), "n_test": len(te)})
                rows.append(r)
            joblib.dump({"model": m, "calibrator": cal_m, "thresholds": variants,
                         "features": fs_name}, MOD / f"{fs_name}_{mname}.joblib")
        print(fs_name, "done", flush=True)
        curves[fs_name] = {"thresholds": THRESHOLDS.tolist(), "f1_macro": f1m,
                           "f1_afib": f1a, "recall_afib": rec}
    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage34_results.csv", index=False)
    (MET / "threshold_curves_all.json").write_text(json.dumps(curves), encoding="utf-8")
    add = []
    for fs_name in FEATURE_SETS:
        sub = df[(df.features == fs_name) & (df.threshold_variant == "max_macro")]
        if len(sub):
            b = sub.sort_values("select_f1", ascending=False).iloc[0]
            add.append(f"- {fs_name}: best select={b['model']} selectF1={b['select_f1']:.4f}, "
                       f"test F1={b['f1_macro']:.4f}, AFIB-F1={b['afib_f1']:.4f}")
    lines = ["# Stage 3.4 — feature-set experiment", "",
             "Все конфигурации на одинаковом 4-way split; пороги — только на threshold split.", "",
             "\n".join(add), "", df.sort_values("select_f1", ascending=False).head(30).to_string(index=False)]
    (ROOT / "reports" / "AF_AFL_STAGE34_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
