"""Stage 3.5 P0 — reproducible baseline on the EXACT Stage 3.4 test split.

Fixes:
- loads the actual Stage 3.4 model bundle (not Stage 3.3);
- reconstructs the identical 4-way split (same seed, same indices);
- writes a single test manifest with record_id/source/patient_key/label/split;
- verifies record_id alignment, prediction count and confusion matrix;
- bootstrap CIs grouped by patient_key (Chapman = record surrogate);
- does NOT tune anything on test and does NOT use test for selection.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage35"
MOD = ROOT / "reports" / "models" / "af_afl_stage34"


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cluster_bootstrap(y, p, thr, groups, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    out = {"f1_macro": [], "afib_f1": [], "afl_f1": []}
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[idx]; pp = p[idx]
        if len(np.unique(yy)) < 2:
            continue
        pr = pp >= thr
        out["f1_macro"].append(f1_score(yy, pr, average="macro"))
        out["afib_f1"].append(f1_score(yy, pr, pos_label=1))
        out["afl_f1"].append(f1_score(yy, pr, pos_label=0))
    return {k: {"mean": float(np.mean(v)), "low": float(np.percentile(v, 2.5)),
                "high": float(np.percentile(v, 97.5))} for k, v in out.items()}


def main() -> int:
    MET.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    y = man.label.to_numpy()
    groups = man.patient_key.to_numpy()
    tr0, held = np.where(sp.split == "train")[0], np.where(sp.split != "train")[0]
    folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42).split(
        held, y[held], groups=groups[held]))
    cal, thr_i, sel, te = [held[folds[i][1]] for i in range(4)]
    assert len(set(te) & set(tr0)) == 0
    assert len(set(te) & set(cal)) == 0 and len(set(te) & set(thr_i)) == 0

    # The model selected by independent validation in Stage 3.4:
    chosen = "all_heat_RandomForest"  # top select_f1 in AF_AFL_STAGE34_REPORT.md
    bundle = joblib_load(MOD / f"{chosen}.joblib")
    feats = bundle["features"]
    files = {"expert": "features_advanced_expert.npy", "v2": "features_advanced_v2.npy",
             "atrial": "features_atrial34.npy", "heat": "features_heat.npy"}
    X = np.concatenate([np.load(D / files[f], mmap_mode="r") for f in ("expert", "v2", "atrial", "heat")], axis=1)
    an = np.isnan(X[tr0]).all(axis=0)
    X = np.delete(X, np.where(an)[0], axis=1)
    med = np.nan_to_num(np.nanmedian(X[tr0], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr0], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    cen = np.median(X[tr0], axis=0)
    X = np.clip((X - cen)/sc, -20, 20).astype(np.float32)
    m, cal_m, variants = bundle["model"], bundle["calibrator"], bundle["thresholds"]
    thr = float(variants["max_macro"])
    p = cal_m.predict_proba(m.predict_proba(X[te])[:, 1].reshape(-1, 1))[:, 1]
    pred = (p >= thr).astype(int)
    yy = y[te]

    # EXACT manifest and predictions
    test_man = man.iloc[te][["record_id", "source", "patient_id", "label"]].copy()
    test_man["patient_key"] = groups[te]
    test_man["split"] = "test"
    test_man["prob_afib"] = p
    test_man["threshold"] = thr
    test_man["prediction"] = pred
    test_man["outcome"] = np.select(
        [(yy == 1) & (pred == 1), (yy == 0) & (pred == 0), (yy == 0) & (pred == 1), (yy == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"], default="?")
    test_man["model"] = chosen
    assert len(test_man) == len(p) == confusion_matrix(yy, pred).sum(), "count mismatch"
    test_man.to_csv(MET / "stage35_test_predictions.csv", index=False)

    ci = cluster_bootstrap(yy, p, thr, groups[te])
    metrics = dict(f1_macro=float(f1_score(yy, pred, average="macro")),
                   afib_f1=float(f1_score(yy, pred, pos_label=1)),
                   afl_f1=float(f1_score(yy, pred, pos_label=0)),
                   afib_recall=float(recall_score(yy, pred, pos_label=1)),
                   afib_precision=float(precision_score(yy, pred, pos_label=1)),
                   afl_recall=float(recall_score(yy, pred, pos_label=0)),
                   accuracy=float(accuracy_score(yy, pred)),
                   balanced_accuracy=float(balanced_accuracy_score(yy, pred)),
                   mcc=float(matthews_corrcoef(yy, pred)),
                   pr_auc_afib=float(average_precision_score(yy, p)),
                   roc_auc=float(roc_auc_score(yy, p)),
                   brier=float(brier_score_loss(yy, p)),
                   log_loss=float(log_loss(yy, np.clip(p, 1e-12, 1-1e-12))),
                   n_test=int(len(te)), threshold=thr, model=chosen,
                   cm=confusion_matrix(yy, pred, labels=[0,1]).tolist(),
                   bootstrap_ci=ci,
                   qc_manifest_sha256=h(D / "qc_final_manifest.csv"),
                   model_sha256=h(MOD / f"{chosen}.joblib"),
                   grouping_note="Chapman: record_id surrogate (no real patient_id); PTB-XL/CPSC: record-level",
                   test_indices_sha256=hashlib.sha256(np.array(te).tobytes()).hexdigest())
    (MET / "stage35_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# Stage 3.5 P0 — достоверный baseline", "",
             f"Модель: **{chosen}** (выбрана по independent validation Stage 3.4).", "",
             f"Тест: {len(te)} записей; record_id проверены; CM = {metrics['cm']}.", "",
             f"F1_macro {metrics['f1_macro']:.4f} CI[{ci['f1_macro']['low']:.4f},{ci['f1_macro']['high']:.4f}]",
             f"AFIB-F1 {metrics['afib_f1']:.4f} CI[{ci['afib_f1']['low']:.4f},{ci['afib_f1']['high']:.4f}]",
             f"AFL-F1 {metrics['afl_f1']:.4f} CI[{ci['afl_f1']['low']:.4f},{ci['afl_f1']['high']:.4f}]", "",
             "Никакие пороги/модели не подбирались на test. Test permutation importance не используется для отбора признаков."]
    (ROOT / "reports" / "AF_AFL_STAGE35_BASELINE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


def joblib_load(p):
    import joblib
    return joblib.load(p)


if __name__ == "__main__":
    raise SystemExit(main())
