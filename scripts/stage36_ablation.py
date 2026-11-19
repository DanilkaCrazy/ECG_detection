"""Stage 3.6.3 — atrial feature ablation on fixed protocol.

Only PREDICTIVE atrial columns (from stage36_3_feature_manifest.json) enter the
classifier; diagnostic quality/status fields never do. Threshold is chosen on
a val half disjoint from calibrator fitting; test is evaluated once.
"""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
D = ROOT / "data" / "processed" / "af_afl_stage34"
D36 = ROOT / "data" / "processed" / "af_afl_stage36"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
THR = np.linspace(0.05, 0.95, 91)
SETS = {
    "A_expert": [],
    "B_expert_median": ["median_"],
    "C_expert_adaptive": ["adaptive_"],
    "D_expert_beat_aligned": ["beat_aligned_"],
    "E_expert_tq": ["tq_segment_"],
    "F_expert_all_atrial": ["median_", "adaptive_", "beat_aligned_", "tq_segment_"],
    "G_expert_selected_atrial": ["__selected__"],
}


def prep(F, tr):
    keep = ~np.isnan(F[tr]).all(axis=0)
    X = np.asarray(F)[:, keep]  # subset FIRST; med then has matching width
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    cen = np.median(X[tr], axis=0)
    return np.clip((X - cen) / sc, -20, 20).astype(np.float32)


def metrics(y, p, thr):
    q = (p >= thr).astype(int)
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 recall_score, precision_score, precision_recall_fscore_support,
                                 confusion_matrix, matthews_corrcoef, brier_score_loss)
    return dict(f1_macro=float(f1_score(y, q, average="macro")),
                afib_f1=float(f1_score(y, q, pos_label=1)),
                afl_f1=float(f1_score(y, q, pos_label=0)),
                afib_recall=float(recall_score(y, q, pos_label=1)),
                afl_recall=float(recall_score(y, q, pos_label=0)),
                afib_precision=float(precision_score(y, q, pos_label=1)),
                roc_auc=float(roc_auc_score(y, p)),
                pr_auc_afib=float(average_precision_score(y, p)),
                mcc=float(matthews_corrcoef(y, q)),
                brier=float(brier_score_loss(y, p)),
                cm=confusion_matrix(y, q, labels=[0, 1]).tolist())


def main():
    manifest = json.loads((MET / "stage36_3_feature_manifest.json").read_text(encoding="utf-8"))
    names_atr = manifest["predictive"]
    atr = np.load(D36 / "features_atrial_residual.npy", mmap_mode="r")
    expert = np.load(D / "features_advanced_expert.npy", mmap_mode="r")
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    y = meta.label.to_numpy()
    tr = np.where(sp.split == "train")[0]
    va = np.where(sp.split == "val")[0]
    te = np.where(sp.split == "test")[0]

    rows = []
    for setname, prefixes in SETS.items():
        if prefixes == ["__selected__"]:
            prefixes = ("median_", "adaptive_", "beat_aligned_", "tq_segment_")
        cols = [i for i, n in enumerate(names_atr)
                if any(n.startswith(p) for p in prefixes)] if prefixes else []
        if setname == "G_expert_selected_atrial" and cols:
            # Top-50 atrial columns selected on TRAIN only
            A = np.asarray(atr[tr][:, cols], dtype=np.float32)
            A = np.nan_to_num(np.where(np.isnan(A), np.nanmedian(A, axis=0), A))
            sel = SelectKBest(f_classif, k=min(50, len(cols)))
            sel.fit(A, y[tr])
            cols = [cols[i] for i in sel.get_support(indices=True)]
        X = np.concatenate([np.asarray(expert, dtype=np.float32),
                            np.asarray(atr[:, cols], dtype=np.float32)], axis=1) if cols \
            else np.asarray(expert, dtype=np.float32)
        Xp = prep(X, tr)
        for mname, m in (("RF", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                       class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
                         ("HistGB", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                                                   max_leaf_nodes=20, min_samples_leaf=25,
                                                                   class_weight="balanced", random_state=42))):
            m.fit(Xp[tr], y[tr])
            sv = m.predict_proba(Xp[va])[:, 1]
            st = m.predict_proba(Xp[te])[:, 1]
            # threshold on val half B, calibrator on val half A
            from sklearn.model_selection import StratifiedGroupKFold
            folds = list(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42).split(
                np.zeros((len(va), 1)), y[va], groups=meta.patient_key.iloc[va].to_numpy()))
            ic, it = folds[0][0], folds[0][1]
            cal = LogisticRegression().fit(sv[ic].reshape(-1, 1), y[va][ic])
            pthr = cal.predict_proba(sv[it].reshape(-1, 1))[:, 1]
            thr = float(THR[int(np.argmax([f1_score(y[va][it], (pthr >= t).astype(int), average="macro")
                                           for t in THR]))])
            pt = cal.predict_proba(st.reshape(-1, 1))[:, 1]
            r = metrics(y[te], pt, thr)
            r.update({"set": setname, "model": mname, "n_atrial_cols": len(cols),
                      "threshold": thr, "n_test": len(te)})
            rows.append(r)
            pd.DataFrame({"record_id": meta.record_id.iloc[te].to_numpy(),
                          "label": y[te], "prob": pt, "prediction": (pt >= thr).astype(int)}).to_csv(
                MET / f"stage36_3_pred_{setname}_{mname}.csv", index=False)
        print(setname, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(MET / "stage36_3_ablation.csv", index=False)
    (MET / "stage36_3_ablation.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage 3.6.3 — Ablation предсердных признаков", "",
             "Только предиктивные колонки; диагностические поля исключены.", "",
             df[["set", "model", "n_atrial_cols", "f1_macro", "afib_f1", "afl_f1",
                 "roc_auc", "pr_auc_afib", "threshold"]].to_string(index=False)]
    (ROOT / "reports" / "AF_AFL_STAGE36_3_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
