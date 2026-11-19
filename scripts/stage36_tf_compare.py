"""Stage 3.6.5 — compare expert vs expert + time-frequency/organization features."""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "data" / "processed" / "af_afl_stage34"
D36 = ROOT / "data" / "processed" / "af_afl_stage36"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
THR = np.linspace(0.05, 0.95, 91)


def prep(F, tr):
    keep = ~np.isnan(F[tr]).all(axis=0)
    X = np.asarray(F)[:, keep]
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[tr], axis=0)) / sc, -20, 20).astype(np.float32)


def main():
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    y = meta.label.to_numpy()
    tr = np.where(sp.split == "train")[0]; va = np.where(sp.split == "val")[0]; te = np.where(sp.split == "test")[0]
    expert = np.load(D / "features_advanced_expert.npy", mmap_mode="r")
    tf = np.load(D36 / "features_temporal_organization.npy", mmap_mode="r")
    rows = []
    for name, F in (("A_expert", np.asarray(expert)), ("H_expert_TF", np.concatenate([np.asarray(expert), np.asarray(tf)], axis=1))):
        X = prep(F, tr)
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=42).fit(X[tr], y[tr])
        sv = m.predict_proba(X[va])[:, 1]; st = m.predict_proba(X[te])[:, 1]
        folds = list(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42).split(
            np.zeros((len(va), 1)), y[va], groups=meta.patient_key.iloc[va].to_numpy()))
        ic, it = folds[0]
        cal = LogisticRegression().fit(sv[ic].reshape(-1, 1), y[va][ic])
        pthr = cal.predict_proba(sv[it].reshape(-1, 1))[:, 1]
        thr = float(THR[int(np.argmax([f1_score(y[va][it], (pthr >= t).astype(int), average="macro") for t in THR]))])
        pt = cal.predict_proba(st.reshape(-1, 1))[:, 1]
        q = (pt >= thr).astype(int)
        r = dict(set=name, n_features=X.shape[1], f1_macro=float(f1_score(y[te], q, average="macro")),
                 afib_f1=float(f1_score(y[te], q, pos_label=1)), afl_f1=float(f1_score(y[te], q, pos_label=0)),
                 roc_auc=float(roc_auc_score(y[te], pt)), pr_auc=float(average_precision_score(y[te], pt)),
                 threshold=thr)
        rows.append(r)
        pd.DataFrame({"record_id": meta.record_id.iloc[te].to_numpy(), "label": y[te],
                      "prob": pt, "prediction": q}).to_csv(MET / f"stage36_5_pred_{name}.csv", index=False)
        print(name, round(r["f1_macro"], 4), round(r["afib_f1"], 4), flush=True)
    (MET / "stage36_5_tf_compare.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
