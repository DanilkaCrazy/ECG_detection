"""Stage 3.6.22 — strict E0/E2/E7/H benchmark on all 7505 v1.1 records.

5 seeds; threshold on Chapman val; evaluation on development TEST only (the
demoted holdout is not used for selection).  Bootstrap resamples BOTH rows and
the seed index, so CIs include training-run variability.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.external_evaluation import evaluate_predictions, select_threshold  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3622"
MET = ROOT / "reports" / "metrics" / "stage3622"
MET.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 13, 21, 99]
SETS = {
    "E0": lambda g: g == "X",
    "E2": lambda g: np.isin(g, ["X", "B"]),
    "E7": lambda g: np.isin(g, ["X", "B", "C"]),
    # Historical compact H = expert 67 + 16 TF/organization features.
    "H": lambda g: (g == "X") | np.array(
        [("org_" in n) or ("stft_" in n) for n in FEATURE_NAMES_HOLDER[0]]),
    "E7_AV": lambda g: np.isin(g, ["X", "B", "C", "AV"]),
}
MODELS = ["RF"]


def prep(X, fit):
    X = np.asarray(X, float)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def model(name, seed):
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                       class_weight="balanced_subsample", n_jobs=-1,
                                       random_state=seed)
    if name == "HistGB":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                              random_state=seed)
    return CatBoostClassifier(iterations=250, depth=5, learning_rate=0.08,
                              verbose=False, thread_count=-1, random_state=seed)


def main() -> int:
    global FEATURE_NAMES_HOLDER
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    fn = pd.read_csv(D / "feature_names.csv")
    names, grp = fn.name.astype(str).to_numpy(), fn.group.astype(str).to_numpy()
    FEATURE_NAMES_HOLDER = [names]
    X = np.load(D / "X.npy", mmap_mode="r")
    y = meta.label.to_numpy().astype(int)
    src, spl = meta.source.to_numpy(), sp.split.to_numpy()
    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    test = np.where((src == "chapman") & (spl == "test"))[0]

    rows, preds = [], {}
    for setname, selector in SETS.items():
        cols = np.where(selector(grp))[0]
        Xp = prep(np.asarray(X[:, cols]), fit)
        for seed in SEEDS:
            for mname in MODELS:
                m = model(mname, seed).fit(Xp[fit], y[fit])
                thr = select_threshold(y[val], m.predict_proba(Xp[val])[:, 1])
                s = m.predict_proba(Xp[test])[:, 1]
                met = evaluate_predictions(y[test], s, thr)
                rows.append({"set": setname, "model": mname, "seed": seed,
                             "n_features": Xp.shape[1], "threshold": thr,
                             **{k: met[k] for k in ("f1_macro", "afib_f1", "afl_f1",
                                                    "roc_auc", "pr_auc", "brier",
                                                    "balanced_accuracy")}})
                preds[(setname, seed)] = {"y": y[test], "p": s, "t": thr,
                                          "keys": meta.patient_key.to_numpy()[test]}
        print(setname, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(MET / "benchmark_results.csv", index=False)
    # Bootstrap rows AND seeds (same seed drawn for all compared models).
    rng = np.random.default_rng(42)
    keys = preds[("E0", 42)]["keys"]
    by = {k: np.where(keys == k)[0] for k in np.unique(keys)}
    boot = {}
    for target in ("E2", "H", "E7_AV", "E7"):
        diffs = []
        for _ in range(1000):
            seed = int(rng.choice(SEEDS))
            idx = np.concatenate([by[k] for k in rng.choice(list(by), len(by), True)])
            def f(name):
                d = preds[(name, seed)]
                return f1_score(d["y"][idx], (d["p"][idx] >= d["t"]).astype(int),
                                average="macro")
            diffs.append(f(target) - f("E0"))
        a = np.asarray(diffs)
        lo, hi = np.percentile(a, [2.5, 97.5])
        boot[target] = {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi),
                        "significant": bool(lo > 0 or hi < 0)}
    # E7 vs E2 with seed-aware CI (the reviewer's key comparison).
    diffs = []
    for _ in range(1000):
        seed = int(rng.choice(SEEDS))
        idx = np.concatenate([by[k] for k in rng.choice(list(by), len(by), True)])
        def f(name):
            d = preds[(name, seed)]
            return f1_score(d["y"][idx], (d["p"][idx] >= d["t"]).astype(int), average="macro")
        diffs.append(f("E7") - f("E2"))
    a = np.asarray(diffs)
    lo, hi = np.percentile(a, [2.5, 97.5])
    boot["E7_minus_E2"] = {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi),
                           "significant": bool(lo > 0 or hi < 0)}

    summary = {"mean_by_set": df.groupby("set")[["f1_macro", "afib_f1", "afl_f1",
                                                "roc_auc", "balanced_accuracy"]].mean().to_dict("index"),
               "bootstrap_vs_E0": boot,
               "n_records": int(len(meta)), "seeds": SEEDS,
               "bootstrap_includes_seed_variability": True}
    (MET / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
