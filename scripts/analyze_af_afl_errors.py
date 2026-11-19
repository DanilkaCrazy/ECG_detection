"""Stage 2 / Error analysis of misclassified AFIB records.

Groups A-E: regular RR, organised atrial activity, poor signal quality,
ambiguous diagnosis, short/noisy RR series. Uses permutation importance
(SHAP is not installed) and PCA instead of UMAP.
"""

from __future__ import annotations

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
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage2"
FIG = ROOT / "reports" / "figures" / "af_afl_stage2"
MOD = ROOT / "reports" / "models" / "af_afl_stage2"
DATASETS = {
    "extended": config.PROCESSED_ROOT / "af_afl_full10s_dedup",
    "strict": config.PROCESSED_ROOT / "af_afl_strict",
}


def features(D, name, tr):
    e = np.load(D / "features_advanced_expert.npy")
    v = np.load(D / "features_advanced_v2.npy")
    h = np.load(D / "features_heat.npy")
    en = pd.read_csv(D / "feature_names_expert.csv")["name"].tolist()
    vn = pd.read_csv(D / "feature_names_v2.csv")["name"].tolist()
    hn = pd.read_csv(D / "feature_names_heat.csv")["name"].tolist()
    X = np.concatenate([e, v, h], axis=1)
    names = [f"expert_{n}" for n in en] + [f"v2_{n}" for n in vn] + [f"heat_{n}" for n in hn]
    an = np.isnan(X[tr]).all(axis=0)
    keep = np.where(~an)[0]
    X = X[:, keep]; names = [names[i] for i in keep]
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    return X, names


def main() -> int:
    results = pd.read_csv(MET / "stage2_results.csv")
    best = results[results.prob_version == "calibrated"].sort_values("f1_macro", ascending=False).iloc[0]
    dname = best["dataset"]
    D = DATASETS[dname]
    pred = pd.read_csv(MET / f"predictions_{dname}_{best['features']}.csv")
    sp = pd.read_csv(D / "split.csv")
    te = sp.loc[sp.split == "test", "idx"].to_numpy()
    tr = sp.loc[sp.split == "train", "idx"].to_numpy()
    yte = np.load(D / "y_test.npy")
    X, names = features(D, best["features"], tr)
    bundle = joblib.load(MOD / f"{dname}_{best['features']}.joblib")
    model, cal = bundle["model"], bundle["calibrator"]
    Xte = X[te]
    p = cal.predict_proba(model.predict_proba(Xte)[:, 1].reshape(-1, 1))[:, 1]
    thr = float(best["threshold"])

    err = pred.copy()
    err["prob"] = p
    err["pred"] = (p >= thr).astype(int)
    err["correct"] = err["y_true"] == err["pred"]
    afib_err = err[(err.y_true == 1) & (~err.correct)].copy()
    afib_ok = err[(err.y_true == 1) & err.correct]
    print(f"AFIB errors: {len(afib_err)} / {int((err.y_true==1).sum())}", flush=True)

    # raw signal QC for error records
    Xraw = np.load(D / "X_test.npy", mmap_mode="r")
    qc = []
    for i in range(len(err)):
        s = np.asarray(Xraw[i], dtype=np.float32)
        qc.append({"std": float(np.std(s)), "maxabs": float(np.max(np.abs(s))),
                   "nan_frac": float(np.isnan(s).mean())})
    qcdf = pd.DataFrame(qc)
    err = pd.concat([err.reset_index(drop=True), qcdf], axis=1)
    afib_err = err[(err.y_true == 1) & (~err.correct)].copy()

    # Assign heuristic groups (not mutually exclusive)
    expert_idx = {n: i for i, n in enumerate(names)}
    def col(name, default=np.nan):
        i = expert_idx.get(name)
        return Xte[:, i] if i is not None else np.full(len(Xte), default)
    rr_sdnn = col("expert_A_sdnn")
    n_beats = col("expert_A_n_beats")
    atrial_ac = col("expert_D_atrial_ac_max")
    pos = {rid: i for i, rid in enumerate(err["record_id"])}
    def errcol(vec):
        return np.array([vec[pos[r]] for r in afib_err["record_id"]])
    afib_err["group_A_regular_RR"] = errcol(rr_sdnn) < np.nanpercentile(rr_sdnn, 25)
    afib_err["group_B_atrial_activity"] = errcol(atrial_ac) > np.nanpercentile(atrial_ac, 75)
    afib_err["group_C_low_quality"] = (afib_err["std"] < 0.05) | (afib_err["maxabs"] > 5) | (afib_err["nan_frac"] > 0)
    # ambiguous: AFL code present but not primary, or multiple AFIB/AFL-related codes
    import re
    man = pd.read_csv(ROOT / "reports" / "manifests" / "chapman_manifest.csv").set_index("record")
    def _amb(rec):
        if rec not in man.index:
            return False
        cs = [c for c in re.split(r"[,;\s]+", str(man.loc[rec, "dx_codes"]).strip("[]'\"")) if c]
        return ("164890007" in cs and (not cs or cs[0] != "164890007")) or (
            "164889003" in cs and (not cs or cs[0] != "164889003"))
    afib_err["group_D_ambiguous"] = afib_err["record_id"].apply(_amb)
    afib_err["group_E_few_beats"] = errcol(n_beats) < np.nanpercentile(n_beats, 10)
    afib_err.to_csv(MET / "afib_errors.csv", index=False)

    # probability distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(err[err.y_true == 1]["prob"], bins=30, alpha=0.6, label="true AFIB", color="#e74c3c")
    ax.hist(err[err.y_true == 0]["prob"], bins=30, alpha=0.6, label="true AFL", color="#3498db")
    ax.axvline(thr, color="k", ls="--", label=f"threshold={thr:.2f}")
    ax.set_xlabel("P(AFIB) calibrated"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("Score distribution — Stage2 best")
    fig.tight_layout(); fig.savefig(FIG / "error_prob_distribution.png", dpi=150); plt.close()

    # PCA of correct vs incorrect AFIB
    sub = Xte[err.y_true.to_numpy() == 1]
    labels = err[err.y_true == 1]["correct"].to_numpy()
    pca = PCA(n_components=2, random_state=42).fit(sub)
    Z = pca.transform(sub)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[labels, 0], Z[labels, 1], s=12, alpha=0.6, label="correct AFIB")
    ax.scatter(Z[~labels, 0], Z[~labels, 1], s=18, alpha=0.8, c="red", label="misclassified AFIB")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.legend(); ax.set_title("PCA: AFIB correct vs misclassified")
    fig.tight_layout(); fig.savefig(FIG / "error_pca.png", dpi=150); plt.close()

    # permutation importance on test holdout subset (500 samples)
    idx = np.arange(len(Xte))
    _, sample = train_test_split(idx, train_size=min(150, len(idx)), random_state=42, stratify=yte)
    pi = permutation_importance(model, Xte[sample], yte[sample], scoring="roc_auc",
                                n_repeats=2, random_state=42, n_jobs=1)
    imp = pd.DataFrame({"feature": [names[i] for i in pi.importances_mean.argsort()[::-1][:30]],
                        "mean": np.sort(pi.importances_mean)[::-1][:30]}).sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(imp["feature"], imp["mean"], color="#2c7fb8")
    ax.set_title("Permutation importance (ROC-AUC, test subset)")
    fig.tight_layout(); fig.savefig(FIG / "permutation_importance.png", dpi=150); plt.close()
    imp.to_csv(MET / "permutation_importance.csv", index=False)

    lines = [
        "# Stage 2 — AFIB error analysis", "",
        f"Best model: {dname} / {best['features']}, threshold {thr:.3f}",
        f"Misclassified AFIB: {len(afib_err)} of {int((err.y_true==1).sum())}", "",
        "Heuristic error groups (non-exclusive):",
        f"- A (regular RR, SDNN < p25): {int(afib_err.group_A_regular_RR.sum())}",
        f"- B (organised atrial activity > p75): {int(afib_err.group_B_atrial_activity.sum())}",
        f"- C (low signal quality): {int(afib_err.group_C_low_quality.sum())}",
        f"- D (ambiguous label): {int(afib_err.group_D_ambiguous.sum())}",
        f"- E (few beats < p10): {int(afib_err.group_E_few_beats.sum())}",
        "", "High-confidence AFIB errors (model predicts AFL with p(AFIB) < 0.1):",
    ]
    hc = afib_err[afib_err["prob"] < 0.1].sort_values("prob")
    lines.append(hc[["record_id", "rhythm", "prob", "std", "maxabs"]].head(30).to_string(index=False))
    lines += ["", "Top permutation importance:", imp.to_string(index=False)]
    (MET / "error_analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
