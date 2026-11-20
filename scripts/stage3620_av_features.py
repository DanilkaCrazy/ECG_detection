"""Stage 3.6.20 — AV-conduction features and controlled ablation.

Uses ONLY train/val/development-test splits.  The (demoted) holdout is not
used for model selection here.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402
from ecgdet.av_conduction_model import av_conduction_features, FEATURE_NAMES  # noqa: E402
from ecgdet.external_evaluation import evaluate_predictions, select_threshold  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3614_v1_1"
F16 = ROOT / "data" / "processed" / "af_afl_stage3616"
SRC = ROOT / "data" / "processed" / "af_afl_stage34" / "X.npy"
MET = ROOT / "reports" / "metrics" / "stage3620"
MET.mkdir(parents=True, exist_ok=True)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def one(sig):
    peaks, _ = multi_lead_qrs(sig, 500.0, LEADS)
    rr = np.diff(peaks) / 500.0 if len(peaks) >= 2 else np.array([0.8])
    f = av_conduction_features(rr)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


def prep(X, fit):
    X = np.asarray(X, float)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    prov = pd.read_csv(D / "diagnosis_provenance.csv")
    rows = prov.row_in_stage34.to_numpy(dtype=int)
    Xraw = np.load(SRC, mmap_mode="r")[rows]
    AV = np.asarray(Parallel(n_jobs=8, batch_size=32, verbose=0)(
        delayed(one)(np.asarray(Xraw[i], dtype=np.float32)) for i in range(len(Xraw))),
        dtype=np.float32)
    fn = pd.read_csv(F16 / "feature_names.csv")
    Xf = np.load(F16 / "X.npy", mmap_mode="r")
    grp = fn.group.to_numpy()
    base_cols = np.where(np.isin(grp, ["X", "B", "C"]))[0]

    # Split alignment: F16 rows correspond to V1 (6180), v1_1 has 7505.
    # Use only the overlapping record ids, in F16 order.
    old_meta = pd.read_csv(F16 / "meta.csv")
    keys = set(old_meta.record_id.astype(str))
    keep = np.array([i for i, r in enumerate(meta.record_id.astype(str)) if r in keys], dtype=int)
    key_to_pos = {r: i for i, r in enumerate(old_meta.record_id.astype(str))}
    base_pos = np.array([key_to_pos[r] for r in meta.record_id.iloc[keep].astype(str)], dtype=int)
    y = meta.label.to_numpy()[keep]
    sp = pd.read_csv(D / "split.csv").iloc[keep].reset_index(drop=True)
    spl = sp.split.to_numpy()
    src = meta.source.to_numpy()[keep]

    fit = np.where((src == "chapman") & np.isin(spl, ["train"]))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    test = np.where((src == "chapman") & (spl == "test"))[0]
    if len(fit) < 10 or len(val) < 10 or len(test) < 10:
        raise RuntimeError("insufficient dev split rows after alignment")

    setups = {
        "E7": np.asarray(Xf[base_pos][:, base_cols], dtype=np.float32),
        "E7_AV": np.column_stack([np.asarray(Xf[base_pos][:, base_cols]), AV[keep]]).astype(np.float32),
        "AV_only": AV[keep],
    }
    results, preds = {}, {}
    for name, X in setups.items():
        Xp = prep(X, fit)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=42).fit(Xp[fit], y[fit])
        sv = m.predict_proba(Xp[val])[:, 1]
        thr = select_threshold(y[val], sv)
        st = m.predict_proba(Xp[test])[:, 1]
        met = evaluate_predictions(y[test], st, thr)
        results[name] = {"n_features": X.shape[1], "threshold": thr, **{
            k: met[k] for k in ("f1_macro", "afib_f1", "afl_f1", "roc_auc", "pr_auc")}}
        preds[name] = {"y": y[test], "p": st, "t": thr,
                       "keys": meta.patient_key.to_numpy()[keep][test]}

    rng = np.random.default_rng(42)
    keys = preds["E7"]["keys"]
    by = {k: np.where(keys == k)[0] for k in np.unique(keys)}
    boot = {}
    for name in ("E7_AV", "AV_only"):
        diffs = []
        for _ in range(1000):
            idx = np.concatenate([by[k] for k in rng.choice(list(by), len(by), True)])
            f0 = f1_score(preds["E7"]["y"][idx],
                          (preds["E7"]["p"][idx] >= preds["E7"]["t"]).astype(int),
                          average="macro")
            f1 = f1_score(preds[name]["y"][idx],
                          (preds[name]["p"][idx] >= preds[name]["t"]).astype(int),
                          average="macro")
            diffs.append(f1 - f0)
        a = np.asarray(diffs)
        lo, hi = np.percentile(a, [2.5, 97.5])
        boot[name] = {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi),
                      "significant": bool(lo > 0 or hi < 0)}

    report = {"results": results, "bootstrap_vs_E7": boot,
              "note": "development splits only; demoted holdout not used"}
    (MET / "av_ablation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
