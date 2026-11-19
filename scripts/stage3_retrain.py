"""Stage 3 — RETRAIN all models on corrected Stage 3.0 data.

Data: manifest_stage3.csv (deduplicated, StratifiedGroupKFold, no patient leakage).
Features: expert(67)+v2(615)+heat(9) from af_afl_merged_v2, indexed by orig_idx.

Sets:
  A = Chapman only
  D = Chapman + PTB-XL
  E = all binary-eligible sources (Chapman, PTB-XL, CPSC-extra, SHDB-AF)

Models: HistGB, CatBoost, XGBoost, LightGBM. Optuna search on set A only;
best configs reused for D/E. Threshold and calibration fitted on VAL only,
test used once for final metrics with bootstrap confidence intervals.
"""

from __future__ import annotations

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
import optuna
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage3_retrain"
FIG = ROOT / "reports" / "figures" / "af_afl_stage3_retrain"
MOD = ROOT / "reports" / "models" / "af_afl_stage3_retrain"
for d in (MET, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
optuna.logging.set_verbosity(optuna.logging.WARNING)

THRESHOLDS = np.linspace(0.10, 0.90, 81)

SETS = {
    "A_chapman": ["chapman"],
    "D_chapman_ptbxl": ["chapman", "ptbxl"],
    "E_all_binary": ["chapman", "ptbxl", "cpsc2018_extra", "shdb-af"],
}


def metrics(y, p, thr):
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
            "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))),
            "brier": float(brier_score_loss(y, p))}


def bootstrap_ci(y, p, thr, key, n=600, seed=42):
    rng = np.random.default_rng(seed)
    pred = (p >= thr).astype(int)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metrics(y[idx], p[idx], thr)[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def make_set(sources, man, E, V, H):
    sub = man[man.source.isin(sources)].reset_index(drop=True)
    oi = sub["orig_idx"].to_numpy()
    X = np.concatenate([np.asarray(E[oi]), np.asarray(V[oi]), np.asarray(H[oi])], axis=1)
    y = sub["label"].to_numpy()
    sp = sub["split_stage3"].to_numpy()
    tr, va, te = np.where(sp == "train")[0], np.where(sp == "val")[0], np.where(sp == "test")[0]
    an = np.isnan(X[tr]).all(axis=0)
    X = np.delete(X, np.where(an)[0], axis=1)
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    cen = np.median(X[tr], axis=0)
    X = np.clip((X - cen) / sc, -20, 20).astype(np.float32)
    return X, y, tr, va, te, sub.iloc[va]["source"].to_numpy(), sub.iloc[te]["source"].to_numpy()


def build_model(name, params, ytr):
    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    if name == "HistGB":
        return HistGradientBoostingClassifier(**params, class_weight="balanced",
                                              early_stopping=True, validation_fraction=0.15,
                                              random_state=42)
    if name == "CatBoost":
        return CatBoostClassifier(**params, loss_function="Logloss", eval_metric="AUC",
                                  auto_class_weights="Balanced", verbose=False,
                                  thread_count=8, random_seed=42)
    if name == "XGBoost":
        return XGBClassifier(**params, scale_pos_weight=neg / pos, eval_metric="auc",
                             tree_method="hist", n_jobs=8, early_stopping_rounds=30,
                             random_state=42)
    if name == "LightGBM":
        return LGBMClassifier(**params, class_weight="balanced", n_jobs=8,
                              verbose=-1, random_state=42)
    raise ValueError(name)


def obj_space(name, t):
    if name == "HistGB":
        return {"max_iter": t.suggest_int("max_iter", 150, 400),
                "learning_rate": t.suggest_float("learning_rate", 0.03, 0.12, log=True),
                "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 40),
                "l2_regularization": t.suggest_float("l2_regularization", 0, 5)}
    if name == "CatBoost":
        return {"iterations": t.suggest_int("iterations", 300, 900),
                "depth": t.suggest_int("depth", 4, 7),
                "learning_rate": t.suggest_float("learning_rate", 0.03, 0.12, log=True),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 6)}
    if name == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 700),
                "max_depth": t.suggest_int("max_depth", 3, 6),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "min_child_weight": t.suggest_int("min_child_weight", 5, 20),
                "subsample": t.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0)}
    return {"n_estimators": t.suggest_int("n_estimators", 200, 700),
            "num_leaves": t.suggest_int("num_leaves", 15, 47),
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 10, 50)}


def fit_model(name, params, Xtr, ytr, Xva, yva):
    m = build_model(name, params, ytr)
    if name == "CatBoost":
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=50, verbose=False)
    elif name == "XGBoost":
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        m.fit(Xtr, ytr)
    return m


def domain(s):
    if s == "chapman":
        return "chapman"
    if s == "shdb-af":
        return "shdb"
    return "external"


def main() -> int:
    man = pd.read_csv(ROOT / "reports" / "metrics" / "af_afl_stage3" / "manifest_stage3.csv")
    D = config.PROCESSED_ROOT / "af_afl_merged_v2"
    E = np.load(D / "features_expert.npy", mmap_mode="r")
    V = np.load(D / "features_v2.npy", mmap_mode="r")
    H = np.load(D / "features_heat.npy", mmap_mode="r")
    best_cfg: dict[str, dict] = {}
    results = []

    # ---------- Set A: Optuna search + final fit ----------
    X, y, tr, va, te, sv, st = make_set(SETS["A_chapman"], man, E, V, H)
    print(f"Set A: train={len(tr)} val={len(va)} test={len(te)} features={X.shape[1]}", flush=True)
    if os.environ.get("REUSE_CFG"):
        best_cfg.update({
            "HistGB": {'max_iter': 237, 'learning_rate': 0.030347133191680812,
                       'max_leaf_nodes': 20, 'min_samples_leaf': 23,
                       'l2_regularization': 0.4447542270401317},
            "CatBoost": {'iterations': 899, 'depth': 6,
                         'learning_rate': 0.030051466063808537,
                         'l2_leaf_reg': 5.446036017127612},
            "XGBoost": {'n_estimators': 409, 'max_depth': 4,
                        'learning_rate': 0.08264910605310478,
                        'min_child_weight': 18, 'subsample': 0.9460313395668616,
                        'colsample_bytree': 0.8815959563095755},
            "LightGBM": {'n_estimators': 206, 'num_leaves': 26,
                         'learning_rate': 0.11718444660738254,
                         'min_child_samples': 31},
        })
        print("reusing saved Optuna configs", flush=True)
    else:
        for name in ("HistGB", "CatBoost", "XGBoost", "LightGBM"):
            def objective(t, nm=name):
                p = obj_space(nm, t)
                m = fit_model(nm, p, X[tr], y[tr], X[va], y[va])
                return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=4)
            best_cfg[name] = study.best_params
            print(f"  {name} best val AUC={study.best_value:.4f} {study.best_params}", flush=True)

    for set_name, sources in SETS.items():
        X, y, tr, va, te, sv, st = make_set(sources, man, E, V, H)
        print(f"\n=== {set_name}: train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)
        for name, params in best_cfg.items():
            m = fit_model(name, params, X[tr], y[tr], X[va], y[va])
            pv = m.predict_proba(X[va])[:, 1]
            # source-specific calibration + threshold on validation
            cals, thrs = {}, {}
            for dom in sorted(set(map(domain, sv))):
                mask = np.array([domain(s) == dom for s in sv])
                if len(np.unique(y[va][mask])) < 2:
                    cals[dom], thrs[dom] = None, 0.5
                    continue
                c = LogisticRegression().fit(pv[mask].reshape(-1, 1), y[va][mask])
                pc = c.predict_proba(pv[mask].reshape(-1, 1))[:, 1]
                sc = [f1_score(y[va][mask], (pc >= t0).astype(int), average="macro") for t0 in THRESHOLDS]
                cals[dom], thrs[dom] = c, float(THRESHOLDS[int(np.argmax(sc))])
            pt = m.predict_proba(X[te])[:, 1]
            for dom in sorted(set(map(domain, st))):
                mask = np.array([domain(s) == dom for s in st])
                if mask.sum() < 5:
                    continue
                c = cals.get(dom)
                thr = thrs.get(dom, 0.5)
                if c is None:
                    c = LogisticRegression().fit(pv.reshape(-1, 1), y[va])
                    thr = 0.5
                p = c.predict_proba(pt[mask].reshape(-1, 1))[:, 1]
                r = metrics(y[te][mask], p, thr)
                lo, hi = bootstrap_ci(y[te][mask], p, thr, "f1_macro")
                r.update({"set": set_name, "model": name, "domain": dom, "threshold": thr,
                          "f1_ci_low": lo, "f1_ci_high": hi, "n_test": int(mask.sum())})
                results.append(r)
                print(f"  {name:10s} {dom:9s} F1={r['f1_macro']:.4f} [{lo:.3f},{hi:.3f}] "
                      f"AFIB={r['afib_f1']:.4f} AUC={r['roc_auc']:.4f}", flush=True)
            joblib.dump({"model": m, "cals": cals, "thrs": thrs, "params": params,
                         "set": set_name, "model_name": name}, MOD / f"{set_name}_{name}.joblib")
            # figures per model on the set (chapman domain if present)
            dom0 = "chapman" if "chapman" in st else sorted(set(map(domain, st)))[0]
            mask = np.array([domain(s) == dom0 for s in st])
            c = cals.get(dom0) or LogisticRegression().fit(pv.reshape(-1, 1), y[va])
            p = c.predict_proba(pt[mask].reshape(-1, 1))[:, 1]
            fpr, tpr, _ = roc_curve(y[te][mask], p); pr, rc, _ = precision_recall_curve(y[te][mask], p)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y[te][mask], p):.3f}")
            ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"ROC {set_name}/{name} ({dom0})")
            fig.tight_layout(); fig.savefig(FIG / f"roc_{set_name}_{name}.png", dpi=150); plt.close()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rc, pr, label=f"AP={average_precision_score(y[te][mask], p):.3f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
            ax.set_title(f"PR {set_name}/{name} ({dom0})")
            fig.tight_layout(); fig.savefig(FIG / f"pr_{set_name}_{name}.png", dpi=150); plt.close()

    df = pd.DataFrame(results)
    df.to_csv(MET / "retrain_results.csv", index=False)
    (MET / "best_configs.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    lines = ["# Stage 3 — переобучение на исправленных данных", "",
             "Данные: `manifest_stage3.csv` (дедупликация, StratifiedGroupKFold, без утечек пациентов).",
             "Порог и калибровка — только на validation, по домену. Тест — один раз.", "",
             df.to_string(index=False), ""]
    ba = df[(df.set == "A_chapman") & (df.domain == "chapman")].sort_values("f1_macro", ascending=False)
    if len(ba):
        best = ba.iloc[0]
        lines += [f"Лучшая модель на Chapman (set A): **{best['model']}** — "
                  f"F1_macro {best['f1_macro']:.4f} [CI {best['f1_ci_low']:.4f}, {best['f1_ci_high']:.4f}], "
                  f"AFIB-F1 {best['afib_f1']:.4f}, AUC {best['roc_auc']:.4f}.", ""]
        cm = None
    lines += ["Сравнение с Stage 2 (0.6094 / 0.4413) не является прямым: другой сплит и другая выборка.",
              "Для объективного сравнения использовать одинаковые фолды (см. AF_AFL_STAGE3_VALIDATION.md)."]
    (ROOT / "reports" / "AF_AFL_STAGE3_RETRAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
