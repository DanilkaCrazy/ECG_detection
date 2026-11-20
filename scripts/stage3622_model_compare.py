"""Stage 3.6.22 — model comparison for E2/E7 (RF, HistGB, CatBoost).

Same strict protocol; threshold on Chapman val, evaluation on development test.
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
SEEDS = [42, 7, 13, 21, 99]


def prep(X, fit):
    X = np.asarray(X, float)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def build(name, seed):
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
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    fn = pd.read_csv(D / "feature_names.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    grp = fn.group.to_numpy()
    y = meta.label.to_numpy().astype(int)
    src, spl = meta.source.to_numpy(), sp.split.to_numpy()
    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    test = np.where((src == "chapman") & (spl == "test"))[0]

    names = ["RF", "HistGB"] + (["CatBoost"] if CatBoostClassifier else [])
    rows, preds = [], {}
    for setname, groups in (("E2", ["X", "B"]), ("E7", ["X", "B", "C"])):
        Xp = prep(np.asarray(X[:, np.isin(grp, groups)]), fit)
        for mname in names:
            for seed in SEEDS:
                m = build(mname, seed).fit(Xp[fit], y[fit])
                thr = select_threshold(y[val], m.predict_proba(Xp[val])[:, 1])
                s = m.predict_proba(Xp[test])[:, 1]
                met = evaluate_predictions(y[test], s, thr)
                rows.append({"set": setname, "model": mname, "seed": seed,
                             **{k: met[k] for k in ("f1_macro", "afib_f1", "afl_f1",
                                                    "roc_auc", "balanced_accuracy")}})
                preds[(setname, mname, seed)] = {"y": y[test], "p": s, "t": thr,
                                                 "keys": meta.patient_key.to_numpy()[test]}
        print(setname, "models done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(MET / "model_comparison.csv", index=False)
    rng = np.random.default_rng(42)
    keys = preds[("E2", "RF", 42)]["keys"]
    by = {k: np.where(keys == k)[0] for k in np.unique(keys)}
    boot = {}
    for mname in names[1:]:
        diffs = []
        for _ in range(1000):
            seed = int(rng.choice(SEEDS))
            idx = np.concatenate([by[k] for k in rng.choice(list(by), len(by), True)])
            def f(k):
                d = preds[k]
                return f1_score(d["y"][idx], (d["p"][idx] >= d["t"]).astype(int),
                                average="macro")
            diffs.append(f(("E2", mname, seed)) - f(("E2", "RF", seed)))
        a = np.asarray(diffs)
        lo, hi = np.percentile(a, [2.5, 97.5])
        boot[mname] = {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi),
                       "significant": bool(lo > 0 or hi < 0)}
    grouped = df.groupby(["set", "model"])[
        ["f1_macro", "afib_f1", "afl_f1", "roc_auc"]].mean()
    summary = {"mean_by_model_set": {
        f"{s}|{m}": v.to_dict() for (s, m), v in grouped.iterrows()},
        "bootstrap_vs_RF_on_E2": boot}
    (MET / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
