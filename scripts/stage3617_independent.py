"""Stage 3.6.17 — frozen-model independent assessment.

1. Sealed holdout: model and threshold frozen on Chapman train/val, then one
   evaluation on the held-out split (never used for selection).
2. Per-source breakdown with bootstrap CIs (patient-level for PTB-XL,
   record-level for Chapman/CPSC — documented explicitly).
3. PTB-XL-only control using official strat_fold 9/10 (human-validated) and
   true patient_id grouping.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.external_evaluation import evaluate_predictions, select_threshold  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3616"
P3614 = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3617"
MET.mkdir(parents=True, exist_ok=True)


def prep(X, fit):
    X = np.asarray(X, dtype=np.float64)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def ci(y, p, t, groups, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    g = np.asarray(groups, dtype=str)
    by = {k: np.where(g == k)[0] for k in np.unique(g)}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[k] for k in rng.choice(list(by), len(by), True)])
        vals.append(f1_score(np.asarray(y)[idx], (np.asarray(p)[idx] >= t).astype(int),
                             average="macro"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean_f1": float(np.mean(vals)), "lo": float(lo), "hi": float(hi)}


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    feat = pd.read_csv(D / "feature_names.csv")
    Xg = np.load(D / "X.npy", mmap_mode="r")
    cols = np.where(np.isin(feat.group.to_numpy(), ["X", "B", "C"]))[0]  # frozen E7
    X = np.asarray(Xg[:, cols])
    y = meta.label.to_numpy().astype(int)
    src = meta.source.to_numpy()
    spl = sp.split.to_numpy()

    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    hold = np.where(spl == "holdout")[0]
    Xp = prep(X, fit)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=42).fit(Xp[fit], y[fit])
    thr = select_threshold(y[val], model.predict_proba(Xp[val])[:, 1])

    result = {"frozen_model": "E7 RF", "threshold": thr, "sealed_holdout": {}}
    sh = model.predict_proba(Xp[hold])[:, 1]
    for name, idx in [("all", np.arange(len(hold))),
                      *[(s, np.where((src[hold] == s))[0]) for s in np.unique(src[hold])]]:
        if len(idx) < 5:
            continue
        m = evaluate_predictions(y[hold][idx], sh[idx], thr)
        block = {k: m[k] for k in ("n", "f1_macro", "afib_f1", "afl_f1",
                                   "roc_auc", "pr_auc", "brier")}
        block["pr_baseline"] = float(y[hold][idx].mean())
        block["bootstrap_ci"] = ci(
            y[hold][idx], sh[idx], thr, meta.patient_key.to_numpy()[hold][idx])
        block["bootstrap_level"] = ("patient" if name == "ptbxl" else "record")
        result["sealed_holdout"][name] = block

    # PTB-XL-only official protocol: strat_fold 1-8 -> train, 9-10 -> test.
    prov = pd.read_csv(P3614 / "diagnosis_provenance.csv")
    ptb_rows = np.where((src == "ptbxl"))[0]
    folds = pd.to_numeric(prov.loc[prov.source == "ptbxl", "strat_fold"],
                          errors="coerce").to_numpy()
    patients = prov.loc[prov.source == "ptbxl", "patient_id"].astype(str).to_numpy()
    if len(folds) == len(ptb_rows) and np.isfinite(folds).all():
        local_tr = np.where(folds <= 8)[0]
        local_te = np.where(folds >= 9)[0]
        tr_pat = set(patients[local_tr])
        local_te = np.asarray([i for i in local_te if patients[i] not in tr_pat], dtype=int)
        tr = ptb_rows[local_tr]
        te = ptb_rows[local_te]
        if len(te) > 5 and np.unique(y[te]).size == 2:
            Xf = prep(X, tr)
            m2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                        class_weight="balanced_subsample", n_jobs=-1,
                                        random_state=42).fit(Xf[tr], y[tr])
            s2 = m2.predict_proba(Xf[te])[:, 1]
            t2 = 0.5
            met2 = evaluate_predictions(y[te], s2, t2)
            result["ptbxl_official_folds"] = {
                "n_train": len(tr), "n_test": len(te), "threshold": t2,
                **{k: met2[k] for k in ("f1_macro", "afib_f1", "afl_f1",
                                        "roc_auc", "pr_auc", "brier")},
                "pr_baseline": float(y[te].mean()),
                "bootstrap_ci": ci(y[te], s2, t2, patients[local_te]),
            }
    (MET / "independent_assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sealed_holdout"},
                     indent=2, default=float))
    for k, v in result["sealed_holdout"].items():
        print(k, "F1", round(v["f1_macro"], 4), "AUC", round(v["roc_auc"], 4),
              "CI", [round(v["bootstrap_ci"]["lo"], 3), round(v["bootstrap_ci"]["hi"], 3)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
