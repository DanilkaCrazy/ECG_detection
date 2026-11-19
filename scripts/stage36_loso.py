"""Stage 3.6.4 — LOSO + source-only baseline on the rebuilt dataset."""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
THR = np.linspace(0.05, 0.95, 91)
FEATS = ["features_advanced_expert.npy"]


def prep(X, tr):
    keep = ~np.isnan(X[tr]).all(axis=0)
    med = np.nan_to_num(np.nanmedian(X[tr][:, keep], axis=0))
    X = np.where(np.isnan(X), med, X)[:, keep]
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    cen = np.median(X[tr], axis=0)
    return np.clip((X - cen) / sc, -20, 20).astype(np.float32)


def main():
    meta = pd.read_csv(D / "meta.csv")
    y = meta.label.to_numpy()
    groups = meta.patient_key.to_numpy()
    src = meta.source.to_numpy()
    X = np.concatenate([np.asarray(np.load(D / f, mmap_mode="r"), dtype=np.float32) for f in FEATS], axis=1)
    rows = []
    for held in sorted(set(src)):
        tr = np.where(src != held)[0]
        te = np.where(src == held)[0]
        Xp = prep(X, tr)
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
                                           min_samples_leaf=25, class_weight="balanced", random_state=42)
        m.fit(Xp[tr], y[tr])
        # threshold on a group-held-out slice of TRAIN sources only
        folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42).split(
            Xp[tr], y[tr], groups=groups[tr]))
        sub_tr, sub_thr = folds[0][0], folds[0][1]
        m2 = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
                                            min_samples_leaf=25, class_weight="balanced", random_state=42)
        m2.fit(Xp[tr][sub_tr], y[tr][sub_tr])
        sthr = m2.predict_proba(Xp[tr][sub_thr])[:, 1]
        cal = LogisticRegression().fit(sthr.reshape(-1, 1), y[tr][sub_thr])
        pthr = cal.predict_proba(sthr.reshape(-1, 1))[:, 1]
        thr = float(THR[int(np.argmax([f1_score(y[tr][sub_thr], (pthr >= t).astype(int), average="macro")
                                       for t in THR]))])
        p = cal.predict_proba(m.predict_proba(Xp[te])[:, 1].reshape(-1, 1))[:, 1]
        q = (p >= thr).astype(int)
        rows.append({"held_out_source": str(held), "n_train": len(tr), "n_test": len(te),
                     "afib_in_test": int((y[te] == 1).sum()), "afl_in_test": int((y[te] == 0).sum()),
                     "threshold": thr,
                     "f1_macro": float(f1_score(y[te], q, average="macro")),
                     "afib_f1": float(f1_score(y[te], q, pos_label=1)),
                     "afl_f1": float(f1_score(y[te], q, pos_label=0)),
                     "roc_auc": float(roc_auc_score(y[te], p)),
                     "pr_auc_afib": float(average_precision_score(y[te], p)),
                     "preprocessing_fit_on": "train sources only"})
        print(held, "done", flush=True)

    # Source-only baseline: source label alone -> AFIB/AFL, evaluated on held-out records
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False).fit(src.reshape(-1, 1))
    S = enc.transform(src.reshape(-1, 1))
    sp = pd.read_csv(D / "split.csv")
    te = np.where(sp.split == "test")[0]
    tr = np.where(sp.split != "test")[0]
    sm = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(S[tr], y[tr])
    ps = sm.predict_proba(S[te])[:, 1]
    qs = (ps >= 0.5).astype(int)
    source_only = {"f1_macro": float(f1_score(y[te], qs, average="macro")),
                   "afib_f1": float(f1_score(y[te], qs, pos_label=1)),
                   "afl_f1": float(f1_score(y[te], qs, pos_label=0)),
                   "roc_auc": float(roc_auc_score(y[te], ps)),
                   "note": "source name only; upper bound of dataset shortcut"}

    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage36_4_loso.csv", index=False)
    summary = {"loso": rows, "source_only_baseline": source_only}
    (MET / "stage36_4_loso.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage 3.6.4 — LOSO и source-only baseline", "", df.to_string(index=False), "",
             f"Source-only (имя источника → класс): F1={source_only['f1_macro']:.4f}, "
             f"AFIB-F1={source_only['afib_f1']:.4f}, AFL-F1={source_only['afl_f1']:.4f}.", "",
             "Chapman: patient_id недоступен, group = record_id (surrogate).",
             "Порог выбран на train-источниках, test-источник не участвовал в настройке."]
    (ROOT / "reports" / "AF_AFL_STAGE36_4_DOMAIN_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
