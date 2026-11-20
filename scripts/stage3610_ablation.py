"""Stage 3.6.10 — ablation E0-E8 over the five physiological feature groups.

Protocol (fixed for every experiment):
  fit      = Chapman train
  threshold= Chapman val        (external labels never used for selection)
  control  = Chapman test
  external = PTB-XL / CPSC-extra, frozen threshold
Metrics, per-source breakdown, paired cluster bootstrap (patient_key) and
Benjamini-Hochberg correction are persisted for every run.
"""
from __future__ import annotations

import hashlib
import json
import sys
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import f_classif

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.external_evaluation import (  # noqa: E402
    ExternalTestProtocol, evaluate_predictions, select_threshold,
)
from ecgdet.feature_selection import prune_constants  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
U = ROOT / "data" / "processed" / "af_afl_stage3610"
MET = ROOT / "reports" / "metrics" / "stage3610"
EXP = ROOT / "experiments" / "stage_3_6_10"
MET.mkdir(parents=True, exist_ok=True)
EXP.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "E0": [], "E1": ["A"], "E2": ["B"], "E3": ["C"], "E4": ["D"],
    "E5": ["E"], "E6": ["A", "B"], "E7": ["B", "C"], "E8": ["A", "B", "C", "D", "E"],
}


def models():
    out = {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=42),
        "HistGB": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                 random_state=42),
    }
    if CatBoostClassifier:
        out["CatBoost"] = CatBoostClassifier(iterations=250, depth=5,
                                             learning_rate=0.08, verbose=False,
                                             thread_count=-1, random_state=42)
    if XGBClassifier:
        out["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4,
                                       learning_rate=0.08, subsample=0.9,
                                       eval_metric="logloss", n_jobs=-1,
                                       random_state=42)
    return out


def prep(X, fit_idx):
    X = np.asarray(X, dtype=np.float64)
    keep = ~np.isnan(X[fit_idx]).all(axis=0)
    X = X[:, keep]
    med = np.nan_to_num(np.nanmedian(X[fit_idx], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit_idx], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit_idx], axis=0)) / sc, -20, 20).astype(np.float32), keep


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main() -> int:
    t0 = time.time()
    meta = pd.read_csv(D / "meta.csv")
    split = pd.read_csv(D / "split.csv")
    expert = np.load(D / "features_advanced_expert.npy", mmap_mode="r")
    Xg = np.load(U / "X.npy")
    feat = pd.read_csv(U / "feature_names.csv")
    group = feat.group.to_numpy()

    fit = np.where((meta.source == "chapman") & (meta.split == "train"))[0]
    val = np.where((meta.source == "chapman") & (meta.split == "val"))[0]
    test = np.where((meta.source == "chapman") & (meta.split == "test"))[0]
    y = meta.label.to_numpy().astype(int)

    all_rows, preds = [], {}
    for exp_id, groups in EXPERIMENTS.items():
        cols = np.where(np.isin(group, groups))[0]
        X = np.column_stack([np.asarray(expert), Xg[:, cols]])
        Xp, kept = prep(X, fit)
        # Drop columns that are constant across the FIT rows only.
        keep_fit = [j for j in range(Xp.shape[1]) if np.ptp(Xp[fit, j]) > 1e-9]
        Xp = Xp[:, keep_fit]
        for mname, model in models().items():
            started = time.time()
            p = ExternalTestProtocol(meta, Xp, model).run(
                source="chapman", external_sources=["ptbxl", "cpsc2018_extra"])
            thr = p["threshold"]
            row = {
                "experiment": exp_id, "groups": "+".join(groups) or "expert",
                "model": mname, "n_features": Xp.shape[1], "threshold": thr,
                "fit_seconds": time.time() - started,
            }
            for eval_name, block in [("chapman_test", p["control"]),
                                     *[(k, v) for k, v in p["external"].items()]]:
                if not block:
                    continue
                for key in ("n", "f1_macro", "afib_f1", "afl_f1", "roc_auc", "pr_auc",
                            "balanced_accuracy", "brier"):
                    row[f"{eval_name}_{key}"] = block[key]
            all_rows.append(row)
            preds[(exp_id, mname)] = {
                "y_test": y[test],
                "prob_test": model.predict_proba(Xp[test])[:, 1],
                "keys": meta.patient_key.to_numpy()[test],
                "threshold": thr,
            }
            print(f"{exp_id} {mname:9} F1={row.get('chapman_test_f1_macro', float('nan')):.4f} "
                  f"({time.time()-started:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(MET / "ablation_results.csv", index=False)

    # Paired cluster bootstrap vs E0, per model, on Chapman test.
    rng = np.random.default_rng(42)
    boot = {}
    for mname in sorted({m for _, m in preds}):
        base = preds[("E0", mname)]
        keys = base["keys"]
        groups_by_key = {k: np.where(keys == k)[0] for k in np.unique(keys)}
        diffs = {e: [] for e in EXPERIMENTS if e != "E0"}
        for _ in range(1000):
            sample = np.concatenate([groups_by_key[k] for k in
                                     rng.choice(list(groups_by_key), len(groups_by_key), True)])
            f0 = f1_score(base["y_test"][sample],
                          (base["prob_test"][sample] >= base["threshold"]).astype(int),
                          average="macro")
            for e in diffs:
                pr = preds[(e, mname)]
                fe = f1_score(pr["y_test"][sample],
                              (pr["prob_test"][sample] >= pr["threshold"]).astype(int),
                              average="macro")
                diffs[e].append(fe - f0)
        rows = []
        for e, v in diffs.items():
            arr = np.asarray(v)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            pval = 2 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
            rows.append({"model": mname, "comparison": f"{e}_minus_E0",
                         "mean": float(arr.mean()), "lo": float(lo), "hi": float(hi),
                         "p_value": min(1.0, pval),
                         "significant": bool(lo > 0 or hi < 0)})
        q = bh_adjust([r["p_value"] for r in rows])
        for r, qq in zip(rows, q):
            r["bh_q"] = float(qq)
        boot[mname] = rows
    (MET / "ablation_bootstrap.json").write_text(
        json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")

    # Persist predictions for the best experiment (by Chapman test F1).
    best = df.sort_values("chapman_test_f1_macro", ascending=False).iloc[0]
    key = (best["experiment"], best["model"])
    pr = preds[key]
    exp_dir = EXP / f"{key[0]}_{key[1]}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": meta.loc[test, "record_id"].to_numpy(),
        "source": meta.loc[test, "source"].to_numpy(),
        "y_true": pr["y_test"],
        "y_pred": (pr["prob_test"] >= pr["threshold"]).astype(int),
        "p_afib": pr["prob_test"], "p_afl": 1 - pr["prob_test"],
        "threshold": pr["threshold"], "experiment_id": f"{key[0]}_{key[1]}",
    }).to_csv(exp_dir / "predictions.csv", index=False)
    (exp_dir / "metrics.json").write_text(json.dumps(best.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "n_experiments": len(EXPERIMENTS), "n_models": len({m for _, m in preds}),
        "best": best.to_dict(), "total_seconds": time.time() - t0,
        "bootstrap_significant": {m: [r for r in v if r["significant"]]
                                  for m, v in boot.items()},
    }
    (MET / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
