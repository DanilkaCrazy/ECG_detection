"""Stage 3.6.16 — corrected E0/E2/E7 benchmark on the clean dataset.

Fit = Chapman train, threshold = Chapman val.  Chapman test is the in-domain
control; the sealed holdout is touched ONLY after the model/threshold are
frozen.  External sets are final evaluation only.  Three seeds and paired
cluster bootstrap quantify uncertainty.
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
MET = ROOT / "reports" / "metrics" / "stage3616"
MET.mkdir(parents=True, exist_ok=True)
EXPERIMENTS = {"E0": ["X"], "E2": ["X", "B"], "E7": ["X", "B", "C"],
               "E8": ["X", "A", "B", "C", "D", "E"]}
SEEDS = [42, 7, 13]


def prep(X, fit):
    X = np.asarray(X, dtype=np.float64)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    feat = pd.read_csv(D / "feature_names.csv")
    Xg = np.load(D / "X.npy", mmap_mode="r")
    grp = feat.group.to_numpy()
    y = meta.label.to_numpy().astype(int)
    src = meta.source.to_numpy()
    spl = sp.split.to_numpy()
    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    ctest = np.where((src == "chapman") & (spl == "test"))[0]
    holdout = np.where(spl == "holdout")[0]
    exts = {s: np.where(src == s)[0] for s in ("ptbxl", "cpsc2018_extra")}

    rows, preds = [], {}
    for eid, groups in EXPERIMENTS.items():
        cols = np.where(np.isin(grp, groups))[0]
        Xp = prep(np.asarray(Xg[:, cols]), fit)
        for seed in SEEDS:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                       class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=seed).fit(Xp[fit], y[fit])
            sv = m.predict_proba(Xp[val])[:, 1]
            thr = select_threshold(y[val], sv)
            rec = {"experiment": eid, "seed": seed, "n_features": Xp.shape[1],
                   "threshold": thr}
            for name, idx in [("chapman_test", ctest), ("sealed_holdout", holdout),
                              *exts.items()]:
                s = m.predict_proba(Xp[idx])[:, 1]
                met = evaluate_predictions(y[idx], s, thr)
                rec[f"{name}_f1_macro"] = met["f1_macro"]
                rec[f"{name}_afib_f1"] = met["afib_f1"]
                rec[f"{name}_afl_f1"] = met["afl_f1"]
                rec[f"{name}_roc_auc"] = met["roc_auc"]
                rec[f"{name}_pr_auc"] = met["pr_auc"]
                rec[f"{name}_pr_baseline"] = float(y[idx].mean())
                rec[f"{name}_brier"] = met["brier"]
                if name == "chapman_test":
                    preds[(eid, seed)] = {"y": y[idx], "p": s, "t": thr,
                                          "keys": meta.patient_key.to_numpy()[idx]}
            rows.append(rec)
            print(f"{eid} seed={seed} F1={rec['chapman_test_f1_macro']:.3f} "
                  f"holdout={rec['sealed_holdout_f1_macro']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(MET / "benchmark_results.csv", index=False)

    # Paired cluster bootstrap (record-level for Chapman) across seeds.
    rng = np.random.default_rng(42)
    boot = {}
    base_keys = preds[("E0", 42)]["keys"]
    by_key = {k: np.where(base_keys == k)[0] for k in np.unique(base_keys)}
    for eid in ("E2", "E7", "E8"):
        diffs = []
        for _ in range(1000):
            sidx = np.concatenate([by_key[k] for k in
                                   rng.choice(list(by_key), len(by_key), True)])
            f0 = f1_score(preds[("E0", 42)]["y"][sidx],
                          (preds[("E0", 42)]["p"][sidx] >= preds[("E0", 42)]["t"]).astype(int),
                          average="macro")
            f1 = f1_score(preds[(eid, 42)]["y"][sidx],
                          (preds[(eid, 42)]["p"][sidx] >= preds[(eid, 42)]["t"]).astype(int),
                          average="macro")
            diffs.append(f1 - f0)
        arr = np.asarray(diffs)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        boot[eid] = {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi),
                     "significant": bool(lo > 0 or hi < 0)}
    agg = df.groupby("experiment")[[c for c in df.columns
                                   if c.endswith(("_f1_macro", "_roc_auc", "_pr_auc"))]].mean()
    summary = {"by_experiment_mean": agg.to_dict("index"), "bootstrap_vs_E0": boot,
               "note": "holdout evaluated only after model+threshold frozen; 3 seeds"}
    (MET / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(agg.round(4).to_string())
    print(json.dumps(boot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
