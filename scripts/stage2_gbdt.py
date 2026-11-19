"""Stage 2 main experiment: GBDT comparison, feature selection, calibration,
threshold optimisation and error dump — on extended and STRICT label sets.

External libraries (CatBoost/XGBoost/LightGBM/Optuna/SHAP) are unavailable in
this environment; sklearn HistGradientBoostingClassifier with a randomised
hyperparameter search is used instead (documented in the report).
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
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures" / "af_afl_stage2"
MET = ROOT / "reports" / "metrics" / "af_afl_stage2"
MOD = ROOT / "reports" / "models" / "af_afl_stage2"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

STATE = 42
THRESHOLDS = np.linspace(0.10, 0.90, 81)
DATASETS = {
    "extended": config.PROCESSED_ROOT / "af_afl_full10s_dedup",
    "strict": config.PROCESSED_ROOT / "af_afl_strict",
}


def feature_sets(D: Path):
    e = np.load(D / "features_advanced_expert.npy")
    v = np.load(D / "features_advanced_v2.npy")
    en = pd.read_csv(D / "feature_names_expert.csv")["name"].tolist()
    vn = pd.read_csv(D / "feature_names_v2.csv")["name"].tolist()
    h = np.load(D / "features_heat.npy")
    hn = pd.read_csv(D / "feature_names_heat.csv")["name"].tolist()
    combined = np.concatenate([e, v], axis=1)
    cnames = [f"expert_{n}" for n in en] + [f"v2_{n}" for n in vn]
    comb_heat = np.concatenate([combined, h], axis=1)
    hnames = [f"heat_{n}" for n in hn]
    return {
        "expert67": (e, [f"expert_{n}" for n in en]),
        "combined682": (combined, cnames),
        "combined_heat691": (comb_heat, cnames + hnames),
    }


def clean(X, names, tr):
    an = np.isnan(X[tr]).all(axis=0)
    X = np.delete(X, np.where(an)[0], axis=1)
    names = [n for i, n in enumerate(names) if i not in set(np.where(an)[0].tolist())]
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, med * 0 + X)
    X = np.where(np.isnan(X), med, X)
    return X, names


def evaluate(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "afib_f1": float(f1_score(y_true, pred, pos_label=1)),
        "afl_f1": float(f1_score(y_true, pred, pos_label=0)),
        "afib_recall": float(recall_score(y_true, pred, pos_label=1)),
        "afib_precision": float(precision_score(y_true, pred, pos_label=1)),
        "afl_recall": float(recall_score(y_true, pred, pos_label=0)),
        "afl_precision": float(precision_score(y_true, pred, pos_label=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "pr_auc_afib": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "log_loss": float(log_loss(y_true, np.clip(prob, 1e-12, 1 - 1e-12))),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def main() -> int:
    all_results = []
    rng = np.random.default_rng(STATE)
    # small randomized search space (HistGB)
    configs = []
    for _ in range(8):
        configs.append({
            "max_iter": int(rng.integers(150, 401)),
            "learning_rate": float(rng.choice([0.04, 0.06, 0.08, 0.1])),
            "max_leaf_nodes": int(rng.choice([15, 31, 63])),
            "min_samples_leaf": int(rng.choice([10, 20, 40])),
            "l2_regularization": float(rng.choice([0.0, 1.0, 5.0])),
        })

    for dname, D in DATASETS.items():
        y = np.load(D / "y.npy").astype(int)
        sp = pd.read_csv(D / "split.csv")
        tr, va, te = [sp.loc[sp.split == s, "idx"].to_numpy() for s in ("train", "val", "test")]
        meta = pd.read_csv(D / "meta.csv")
        print(f"\n=== dataset {dname}: n={len(y)}, AFIB={int(y.sum())}, AFL={int((y==0).sum())} ===", flush=True)
        sets = feature_sets(D)
        # add selected-feature variants from combined_heat
        baseX, baseN = sets["combined_heat691"]
        fill = np.nan_to_num(np.nanmedian(baseX[tr], axis=0))
        baseXf = np.where(np.isnan(baseX), fill, baseX)
        for k in (100, 200, 300):
            sel = SelectKBest(f_classif, k=min(k, baseXf.shape[1])).fit(baseXf[tr], y[tr])
            sets[f"top{k}"] = (sel.transform(baseXf), [n for i, n in enumerate(baseN) if sel.get_support()[i]])

        for sname, (Xraw, names) in sets.items():
            X, names = clean(Xraw, names, tr)
            Xtr, Xva, Xte = X[tr], X[va], X[te]
            ytr, yva, yte = y[tr], y[va], y[te]
            best = None
            t0 = time.time()
            for cfg in configs:
                model = HistGradientBoostingClassifier(**cfg, class_weight="balanced",
                                                       early_stopping=True, validation_fraction=0.15,
                                                       random_state=STATE)
                model.fit(Xtr, ytr)
                pva = model.predict_proba(Xva)[:, 1]
                auc = roc_auc_score(yva, pva)
                if best is None or auc > best[0]:
                    best = (auc, cfg, model)
            auc_va, cfg, model = best
            pva = model.predict_proba(Xva)[:, 1]
            pte = model.predict_proba(Xte)[:, 1]
            # threshold on validation, calibration on validation (sigmoid 1D)
            scores = [f1_score(yva, (pva >= t).astype(int), average="macro") for t in THRESHOLDS]
            thr = float(THRESHOLDS[int(np.argmax(scores))])
            cal = LogisticRegression().fit(pva.reshape(-1, 1), yva)
            pte_cal = cal.predict_proba(pte.reshape(-1, 1))[:, 1]
            scores_cal = [f1_score(yva, (cal.predict_proba(pva.reshape(-1, 1))[:, 1] >= t).astype(int),
                                   average="macro") for t in THRESHOLDS]
            thr_cal = float(THRESHOLDS[int(np.argmax(scores_cal))])
            for p, t, tag in ((pte, thr, "raw"), (pte_cal, thr_cal, "calibrated")):
                r = evaluate(yte, p, t)
                r.update({"dataset": dname, "features": sname, "cfg": json.dumps(cfg),
                          "val_auc": float(auc_va), "threshold": t, "prob_version": tag,
                          "time": time.time() - t0})
                all_results.append(r)
            print(f"  {sname:18s}: F1={all_results[-2]['f1_macro']:.4f} AFIB-F1={all_results[-2]['afib_f1']:.4f} "
                  f"AUC={all_results[-2]['roc_auc']:.4f} calF1={all_results[-1]['f1_macro']:.4f}", flush=True)
            # save best model per feature set
            joblib.dump({"model": model, "calibrator": cal, "threshold": thr, "threshold_cal": thr_cal,
                         "features": sname, "dataset": dname, "cfg": cfg, "names": names},
                        MOD / f"{dname}_{sname}.joblib")
            # dump predictions for error analysis (calibrated best)
            pred_df = meta.iloc[te][["record_id", "rhythm"]].copy()
            pred_df["y_true"] = yte
            pred_df["prob_afib"] = pte_cal
            pred_df["pred"] = (pte_cal >= thr_cal).astype(int)
            pred_df.to_csv(MET / f"predictions_{dname}_{sname}.csv", index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(MET / "stage2_results.csv", index=False)
    (MET / "stage2_results_full.txt").write_text(df.to_string(index=False), encoding="utf-8")

    # comparison figures
    for metric in ("f1_macro", "afib_f1", "afib_recall", "pr_auc_afib", "roc_auc", "log_loss"):
        pv = df[df.prob_version == "calibrated"].pivot_table(index="features", columns="dataset", values=metric)
        pv.plot(kind="bar", figsize=(11, 5), title=f"Stage2: {metric}")
        plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        plt.savefig(FIG / f"stage2_{metric}.png", dpi=150); plt.close()

    # ROC/PR/calibration/CM for global best
    best = df[df.prob_version == "calibrated"].sort_values("f1_macro", ascending=False).iloc[0]
    print("\nBEST:", best.to_dict(), flush=True)
    d = DATASETS[best["dataset"]]
    bundle = joblib.load(MOD / f"{best['dataset']}_{best['features']}.joblib")
    sp = pd.read_csv(d / "split.csv")
    te = sp.loc[sp.split == "test", "idx"].to_numpy()
    yte = np.load(d / "y_test.npy")
    X, names = clean(*feature_sets(d)[best["features"]], sp.loc[sp.split == "train", "idx"].to_numpy())
    p = bundle["calibrator"].predict_proba(bundle["model"].predict_proba(X[te])[:, 1].reshape(-1, 1))[:, 1]
    thr = float(best["threshold"])
    fpr, tpr, _ = roc_curve(yte, p); pr, rc, _ = precision_recall_curve(yte, p)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={best['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title("Stage2 best ROC")
    fig.tight_layout(); fig.savefig(FIG / "best_roc.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(rc, pr, label=f"AP={best['pr_auc_afib']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title("Stage2 best PR")
    fig.tight_layout(); fig.savefig(FIG / "best_pr.png", dpi=150); plt.close()
    cm = confusion_matrix(yte, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AFL", "AFIB"],
                yticklabels=["AFL", "AFIB"], ax=ax)
    ax.set_title(f"Stage2 best CM F1={best['f1_macro']:.3f}")
    fig.tight_layout(); fig.savefig(FIG / "best_cm.png", dpi=150); plt.close()
    # calibration curve
    from sklearn.calibration import calibration_curve
    frac, mean = calibration_curve(yte, p, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(mean, frac, marker="o", label="model")
    ax.plot([0, 1], [0, 1], "k--", label="perfect"); ax.legend()
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("True fraction")
    ax.set_title("Calibration curve — Stage2 best")
    fig.tight_layout(); fig.savefig(FIG / "best_calibration.png", dpi=150); plt.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
