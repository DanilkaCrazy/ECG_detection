"""Stage 3.6.2 — baseline retraining on the REBUILT dataset.

Strict protocol: train fits model+preprocessing; val calibrates and selects the
threshold; test is evaluated once. No test-based tuning, no test predictions
used for feature/model choice.
"""
from __future__ import annotations
import json, warnings, pathlib, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from catboost import CatBoostClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
MOD = ROOT / "reports" / "models" / "af_afl_stage36"
MET.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
THR = np.linspace(0.05, 0.95, 91)


def metrics(y, p, thr):
    q = (p >= thr).astype(int)
    return dict(f1_macro=float(f1_score(y, q, average="macro")),
                afib_f1=float(f1_score(y, q, pos_label=1)),
                afl_f1=float(f1_score(y, q, pos_label=0)),
                afib_recall=float(recall_score(y, q, pos_label=1)),
                afl_recall=float(recall_score(y, q, pos_label=0)),
                afib_precision=float(precision_score(y, q, pos_label=1)),
                accuracy=float(((q == y).mean())),
                balanced_accuracy=float(balanced_accuracy_score(y, q)),
                mcc=float(matthews_corrcoef(y, q)),
                pr_auc_afib=float(average_precision_score(y, p)),
                roc_auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, np.clip(p, 1e-12, 1-1e-12))),
                cm=confusion_matrix(y, q, labels=[0, 1]).tolist())


def calibrate(yv, score_v, yt, score_t):
    cal = LogisticRegression().fit(score_v.reshape(-1, 1), yv)
    pv = cal.predict_proba(score_v.reshape(-1, 1))[:, 1]
    thr = float(THR[int(np.argmax([f1_score(yv, (pv >= t).astype(int), average="macro") for t in THR]))])
    pt = cal.predict_proba(score_t.reshape(-1, 1))[:, 1]
    return cal, thr, pt


def main():
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    y = meta.label.to_numpy()
    tr = np.where(sp.split == "train")[0]
    va = np.where(sp.split == "val")[0]
    te = np.where(sp.split == "test")[0]
    print(f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    F = np.asarray(np.load(D / "features_advanced_expert.npy", mmap_mode="r"), dtype=np.float32)
    keep = ~np.isnan(F[tr]).all(axis=0)
    med = np.nan_to_num(np.nanmedian(F[tr][:, keep], axis=0))
    F = np.where(np.isnan(F), med, F)[:, keep]
    q1, q3 = np.percentile(F[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    cen = np.median(F[tr], axis=0)
    F = np.clip((F - cen) / sc, -20, 20).astype(np.float32)
    np.save(MET / "stage36_2_preprocessing.npy", np.vstack([keep, med, sc, cen]))

    models = {
        "expert_RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "expert_HistGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=20, min_samples_leaf=25, l2_regularization=0.5, class_weight="balanced", random_state=42),
        "expert_CatBoost": CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3, loss_function="Logloss", auto_class_weights="Balanced", verbose=False, thread_count=8, random_seed=42),
    }
    rows = []
    for name, m in models.items():
        m.fit(F[tr], y[tr])
        sv = m.predict_proba(F[va])[:, 1]
        st = m.predict_proba(F[te])[:, 1]
        cal, thr, pt = calibrate(y[va], sv, y[te], st)
        r = metrics(y[te], pt, thr)
        r.update({"model": name, "threshold": float(thr), "n_test": len(te), "calibration": "logistic on val"})
        rows.append(r)
        joblib.dump({"model": m, "calibrator": cal, "threshold": thr}, MOD / f"{name}.joblib")
        pd.DataFrame({"record_id": meta.record_id.iloc[te].to_numpy(), "source": meta.source.iloc[te].to_numpy(),
                      "label": y[te], "prob_afib": pt, "threshold": thr,
                      "prediction": (pt >= thr).astype(int)}).to_csv(MET / f"stage36_2_pred_{name}.csv", index=False)
        print(name, "F1", round(r["f1_macro"], 4), "AFIB", round(r["afib_f1"], 4), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage36_2_baseline.csv", index=False)
    (MET / "stage36_2_baseline.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage 3.6.2 — новый baseline на пересобранных данных", "",
             f"train={len(tr)}, val={len(va)} (порог и калибровка), test={len(te)} (одна оценка).", "",
             "Только expert-признаки (67), без новых atrial-признаков.", "",
             df[["model", "f1_macro", "afib_f1", "afl_f1", "afib_recall", "roc_auc", "pr_auc_afib", "brier", "threshold"]].to_string(index=False)]
    (ROOT / "reports" / "AF_AFL_STAGE36_2_BASELINE.md").write_text("\n".join(lines), encoding="utf-8")
    print("CLASSICAL DONE", flush=True)


if __name__ == "__main__":
    main()
