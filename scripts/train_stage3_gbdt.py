"""Train CatBoost / XGBoost / LightGBM with Optuna on merged_v2.

Features: aligned expert + v2 + heat (691). Evaluation on untouched tests:
Chapman, external PTB-XL/CPSC, SHDB-AF. SHAP analysis for the best model.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_merged_v2"
FIG = ROOT / "reports" / "figures" / "af_afl_stage3"
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
MOD = ROOT / "reports" / "models" / "af_afl_stage3"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    e = np.load(D / "features_expert.npy")
    v = np.load(D / "features_v2.npy")
    h = np.load(D / "features_heat.npy")
    en = pd.read_csv(D / "feature_names_expert.csv")["name"].tolist()
    vn = pd.read_csv(D / "feature_names_v2.csv")["name"].tolist()
    hn = pd.read_csv(D / "feature_names_heat.csv")["name"].tolist()
    X = np.concatenate([e, v, h], axis=1)
    names = en + vn + hn
    X = np.nan_to_num(X, nan=-9.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    y = np.load(D / "y.npy").astype(int)
    sp = pd.read_csv(D / "split.csv")
    idx = {s: sp.loc[sp.split == s, "idx"].to_numpy() for s in
           ("train", "val", "test_chapman", "test_external", "test_shdb")}
    tr, va = idx["train"], idx["val"]
    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
    print(f"Merged_v2: train={len(tr)} AFIB={int(ytr.sum())}, val={len(va)}, features={X.shape[1]}", flush=True)

    scale_pos = float((ytr == 0).sum() / (ytr == 1).sum())
    results = []

    mva = pd.read_csv(D / "meta_val.csv")["source"].to_numpy()
    pva_raw: np.ndarray | None = None
    def dom_of(test):
        return {"test_chapman": "chapman", "test_external": "external", "test_shdb": "shdb"}[test]
    def calib_for(src):
        m = (np.array([s == "chapman" for s in mva]) if src == "chapman" else
             np.array([s == "shdb-af" for s in mva]) if src == "shdb" else
             np.array([s in ("ptbxl", "cpsc2018", "cpsc2018_extra") for s in mva]))
        if m.sum() == 0 or len(np.unique(yva[m])) < 2:
            return None, None
        c = LogisticRegression().fit(pva_raw.reshape(-1, 1)[m], yva[m])
        pc = c.predict_proba(pva_raw.reshape(-1, 1)[m])[:, 1]
        sc = [f1_score(yva[m], (pc >= t).astype(int), average="macro") for t in THRESHOLDS]
        return c, float(THRESHOLDS[int(np.argmax(sc))])

    def calibrate(model, name, kind):
        nonlocal pva_raw
        pva_raw = model.predict_proba(Xva)[:, 1]
        cs, ts = {}, {}
        for src in ("chapman", "external", "shdb"):
            c, t = calib_for(src)
            if c is not None:
                cs[src], ts[src] = c, t
        for test in ("test_chapman", "test_external", "test_shdb"):
            dom = dom_of(test)
            c = cs.get(dom); thr = ts.get(dom, 0.5)
            if c is None:
                c = LogisticRegression().fit(pva_raw.reshape(-1, 1), yva)
                thr = 0.5
            ids = idx[test]
            p = c.predict_proba(model.predict_proba(X[ids])[:, 1].reshape(-1, 1))[:, 1]
            r = evaluate(y[ids], p, thr)
            r.update({"model": name, "test": test, "threshold": thr,
                      "val_auc": roc_auc_score(yva, pva_raw)})
            results.append(r)
            print(f"  {name:14s} {test}: F1={r['f1_macro']:.4f} AFIB={r['afib_f1']:.4f} "
                  f"AUC={r['roc_auc']:.4f}", flush=True)
        return model, cs.get("chapman"), ts.get("chapman", 0.5)

    # ---------- CatBoost with Optuna (short search) ----------
    def cb_objective(t):
        params = {"iterations": t.suggest_int("iterations", 300, 1200),
                  "depth": t.suggest_int("depth", 4, 8),
                  "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                  "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10),
                  "loss_function": "Logloss", "eval_metric": "AUC",
                  "auto_class_weights": "Balanced", "verbose": False,
                  "thread_count": 8, "random_seed": 42}
        m = CatBoostClassifier(**params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=60, verbose=False)
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    study = optuna.create_study(direction="maximize")
    study.optimize(cb_objective, n_trials=5)
    print("CatBoost best:", study.best_params, flush=True)
    cb = CatBoostClassifier(**study.best_params, verbose=False, random_seed=42)
    cb.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=60, verbose=False)
    cb, cal_cb, thr_cb = calibrate(cb, "CatBoost", "cat")

    # ---------- XGBoost ----------
    def xgb_objective(t):
        params = {"n_estimators": t.suggest_int("n_estimators", 200, 800),
                  "max_depth": t.suggest_int("max_depth", 3, 8),
                  "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                  "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
                  "subsample": t.suggest_float("subsample", 0.6, 1.0),
                  "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                  "scale_pos_weight": scale_pos,
                  "eval_metric": "auc", "tree_method": "hist", "n_jobs": 8,
                  "early_stopping_rounds": 40, "random_state": 42}
        m = XGBClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    study2 = optuna.create_study(direction="maximize")
    study2.optimize(xgb_objective, n_trials=5)
    print("XGBoost best:", study2.best_params, flush=True)
    xg = XGBClassifier(**study2.best_params, eval_metric="auc", tree_method="hist",
                       n_jobs=8, early_stopping_rounds=40, random_state=42)
    xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    xg, cal_xg, thr_xg = calibrate(xg, "XGBoost", "xgb")

    # ---------- LightGBM ----------
    def lgb_objective(t):
        params = {"n_estimators": t.suggest_int("n_estimators", 200, 900),
                  "num_leaves": t.suggest_int("num_leaves", 15, 63),
                  "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                  "min_child_samples": t.suggest_int("min_child_samples", 10, 60),
                  "class_weight": "balanced", "n_jobs": 8, "verbose": -1,
                  "random_state": 42}
        m = LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[])
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    study3 = optuna.create_study(direction="maximize")
    study3.optimize(lgb_objective, n_trials=5)
    print("LightGBM best:", study3.best_params, flush=True)
    lg = LGBMClassifier(**study3.best_params, class_weight="balanced", n_jobs=8,
                        verbose=-1, random_state=42)
    lg.fit(Xtr, ytr)
    lg, cal_lg, thr_lg = calibrate(lg, "LightGBM", "lgb")

    df = pd.DataFrame(results)
    df.to_csv(MET / "stage3_results.csv", index=False)
    (MET / "stage3_report.txt").write_text(df.to_string(index=False), encoding="utf-8")

    # figures per model/test
    for name, model in (("CatBoost", cb), ("XGBoost", xg), ("LightGBM", lg)):
        ids = idx["test_chapman"]
        p = model.predict_proba(X[ids])[:, 1]
        fpr, tpr, _ = roc_curve(y[ids], p); pr, rc, _ = precision_recall_curve(y[ids], p)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"ROC — {name} (Chapman test)")
        fig.tight_layout(); fig.savefig(FIG / f"roc_{name}.png", dpi=150); plt.close()

    # SHAP for best model on Chapman test subset
    best = df[df.test == "test_chapman"].sort_values("f1_macro", ascending=False).iloc[0]
    bmodel = {"CatBoost": cb, "XGBoost": xg, "LightGBM": lg}[best["model"]]
    ids = idx["test_chapman"]
    Xs = X[ids][:300]
    explainer = shap.TreeExplainer(bmodel)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list): sv = sv[1]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, Xs, feature_names=names, max_display=25, show=False)
    plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=130); plt.close()
    joblib.dump({"model": bmodel, "best": best.to_dict(), "names": names}, MOD / "stage3_best.joblib")
    print("\nBEST:", best.to_dict(), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
