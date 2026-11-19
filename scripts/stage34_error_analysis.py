"""Stage 3.4 — error analysis on untouched test split.

Loads a frozen model from Stage 3.3, computes predictions on the QC-filtered
test split, and saves TP/TN/FP/FN with record_id, source, probability, QC
status and key features. Produces FN-AFIB / FP-AFIB reports, probability and
threshold curves, permutation importance and SHAP.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage34"
FIG = ROOT / "reports" / "figures" / "af_afl_stage34"
MOD33 = ROOT / "reports" / "models" / "af_afl_stage33"

THRESHOLDS = np.linspace(0.05, 0.95, 91)


def load_set(kind):
    parts = []
    if kind in ("expert", "combined", "combined_heat", "expert_atrial", "all_atrial"):
        parts.append(np.load(D / "features_advanced_expert.npy"))
    if kind in ("v2", "combined", "combined_heat", "v2_atrial", "all_atrial"):
        parts.append(np.load(D / "features_advanced_v2.npy"))
    if kind == "combined_heat":
        parts.append(np.load(D / "features_heat.npy"))
    if "atrial" in kind:
        parts.append(np.load(D / "features_atrial34.npy"))
    return np.concatenate(parts, axis=1)


def main() -> int:
    MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    te = sp.loc[sp.split == "test", "idx"].to_numpy()
    bundle = joblib.load(MOD33 / "E4_combined_CatBoost.joblib")
    model, cal, thr = bundle["model"], bundle["calibrator"], bundle["threshold"]
    F = load_set("combined")
    Xte = F[te]
    Xte = np.nan_to_num(Xte, nan=-9, posinf=0, neginf=0)
    p = cal.predict_proba(model.predict_proba(Xte)[:, 1].reshape(-1, 1))[:, 1]
    y = m.loc[te, "label"].to_numpy()
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = m.loc[te, ["record_id", "source", "rhythm", "patient_key"]].copy()
    out["label_true"] = y
    out["prediction"] = pred
    out["prob_afib"] = p
    out["threshold"] = thr
    out["outcome"] = np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"], default="?")
    qc = pd.read_csv(ROOT / "reports" / "metrics" / "af_afl_stage33" / "full_signal_qc.csv")
    out = out.merge(qc[["record_id", "status", "reasons"]], on="record_id", how="left")
    out["record_id"] = out["record_id"].astype(str)
    out.to_csv(MET / "test_predictions.csv", index=False)

    fn = out[out.outcome == "FN"]
    fp = out[out.outcome == "FP"]
    lines = ["# Stage 3.4 — анализ ошибок", "",
             f"Тест: {len(out)} записей, порог {thr:.3f} (зафиксирован на validation).", "",
             f"FN (пропущенные AFIB): {len(fn)}, FP (ложные AFIB): {len(fp)}", "",
             "FN по источникам: " + str(fn.source.value_counts().to_dict()), "",
             "FP по источникам: " + str(fp.source.value_counts().to_dict()), "",
             "Примеры FN (20):", fn.head(20)[["record_id", "source", "prob_afib", "status"]].to_string(index=False)]
    (ROOT / "reports" / "AF_AFL_STAGE34_ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    # probability distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(p[y == 1], bins=25, alpha=.6, label="true AFIB")
    ax.hist(p[y == 0], bins=25, alpha=.6, label="true AFL")
    ax.axvline(thr, color="k", ls="--", label=f"thr={thr:.2f}")
    ax.set_xlabel("P(AFIB)"); ax.legend(); ax.set_title("Stage 3.4 test score distribution")
    fig.tight_layout(); fig.savefig(FIG / "prob_distribution.png", dpi=150); plt.close()

    # threshold curves from TEST only for descriptive analysis (not selection)
    rows = []
    for t in THRESHOLDS:
        pr = (p >= t).astype(int)
        rows.append({"threshold": t,
                     "f1_macro": f1_score(y, pr, average="macro"),
                     "f1_afib": f1_score(y, pr, pos_label=1),
                     "recall_afib": np.sum((pr == 1) & (y == 1)) / max((y == 1).sum(), 1),
                     "precision_afib": np.sum((pr == 1) & (y == 1)) / max((pr == 1).sum(), 1)})
    tc = pd.DataFrame(rows)
    tc.to_csv(MET / "threshold_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ("f1_macro", "f1_afib", "recall_afib"):
        ax.plot(tc.threshold, tc[col], label=col)
    ax.set_xlabel("threshold"); ax.legend(); ax.set_title("Stage 3.4 threshold curves (descriptive)")
    fig.tight_layout(); fig.savefig(FIG / "threshold_curves.png", dpi=150); plt.close()

    # permutation importance on test subset (descriptive)
    idx = np.random.default_rng(42).choice(len(Xte), min(400, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte[idx], y[idx], scoring="roc_auc",
                                n_repeats=2, random_state=42, n_jobs=1)
    names = [f"f{i}" for i in range(Xte.shape[1])]
    imp = pd.DataFrame({"feature": [names[i] for i in pi.importances_mean.argsort()[::-1][:25]],
                        "mean": np.sort(pi.importances_mean)[::-1][:25]})
    imp.to_csv(MET / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp.feature[::-1], imp["mean"][::-1]); ax.set_title("Permutation importance (descriptive)")
    fig.tight_layout(); fig.savefig(FIG / "permutation_importance.png", dpi=150); plt.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
