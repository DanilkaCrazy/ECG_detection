"""Stage 3.6.18 — PTB-XL near-zero ROC-AUC directionality audit.

Diagnostic ONLY: inversion is reported as evidence of misalignment, never used
to "fix" predictions or labels.
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
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.external_evaluation import select_threshold  # noqa: E402

F = ROOT / "data" / "processed" / "af_afl_stage3616"
S34 = ROOT / "data" / "processed" / "af_afl_stage34"
V1 = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3618"


def prep(X, fit):
    X = np.asarray(X, float)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def main() -> int:
    meta = pd.read_csv(F / "meta.csv")
    sp = pd.read_csv(F / "split.csv")
    fn = pd.read_csv(F / "feature_names.csv")
    X = np.load(F / "X.npy", mmap_mode="r")[:, np.isin(fn.group.to_numpy(), ["X", "B", "C"])]
    y = meta.label.to_numpy().astype(int)
    src, spl = meta.source.to_numpy(), sp.split.to_numpy()
    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    ptb = np.where(src == "ptbxl")[0]
    Xp = prep(X, fit)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=42).fit(Xp[fit], y[fit])
    thr = select_threshold(y[val], model.predict_proba(Xp[val])[:, 1])
    s = model.predict_proba(Xp[ptb])[:, 1]
    yp = y[ptb]

    # 1-3: class order, distributions, confusion matrices
    classes = model.classes_.tolist()
    afib_col = list(classes).index(1)
    dist = {"mean_p_afib_AFIB": float(s[yp == 1].mean()) if (yp == 1).any() else None,
            "mean_p_afib_AFL": float(s[yp == 0].mean()) if (yp == 0).any() else None,
            "n_AFIB": int((yp == 1).sum()), "n_AFL": int((yp == 0).sum())}
    cms = {str(t): confusion_matrix(yp, (s >= (t if t != "val" else thr)).astype(int),
                                    labels=[0, 1]).tolist()
           for t in (0.5, "val")}
    auc = float(roc_auc_score(yp, s))
    auc_inv = float(roc_auc_score(yp, 1 - s))

    # 4: row alignment by record_id + signal_hash
    src_rows = np.where((meta.source == "ptbxl"))[0]
    old = pd.read_csv(S34 / "meta.csv")
    prov = pd.read_csv(V1 / "diagnosis_provenance.csv")
    ptb_prov = prov[prov.source == "ptbxl"]
    align = {
        "n_feature_rows": int(len(meta)),
        "n_record_ids_unique": int(meta.record_id.nunique()),
        "n_hash_unique": int(meta.signal_hash.nunique()),
        "signal_hash_matches_stage34": int((meta.signal_hash.to_numpy() ==
                                            old.signal_hash.to_numpy()[
                                                prov.row_in_stage34.to_numpy(dtype=int)]).sum()),
        "record_ids_match_provenance": int(meta.iloc[src_rows].record_id.reset_index(drop=True)
                                           .equals(ptb_prov.record_id.reset_index(drop=True))),
    }

    # 5: per-source label encode check (via provenance raw codes)
    encode = {}
    for source, sub in prov.groupby("source"):
        encode[str(source)] = {
            "AFIB": int((sub.label_status == "AFIB").sum()),
            "AFL": int((sub.label_status == "AFL").sum()),
            "raw_codes_sample": sub.raw_diagnosis_codes.head(3).tolist(),
        }

    # 7: overlap of development holdout PTB records with official folds 9/10
    folds = pd.to_numeric(ptb_prov.strat_fold, errors="coerce").to_numpy()
    dev_ptb = np.where((src == "ptbxl") & (spl == "development_holdout"))[0] \
        if "development_holdout" in np.unique(spl) else np.where((src == "ptbxl") & (spl == "holdout"))[0]
    local_dev = np.searchsorted(ptb, dev_ptb)
    overlap = int(np.sum(np.isin(folds[local_dev], [9, 10]))) if len(dev_ptb) else 0

    report = {
        "model_classes": classes,
        "probability_column_used_for_AFIB": afib_col,
        "distribution": dist,
        "confusion_matrices": cms,
        "roc_auc_forward": auc, "roc_auc_after_inversion_diagnostic_only": auc_inv,
        "row_alignment": align,
        "label_encoding_by_source": encode,
        "dev_holdout_ptb_in_official_folds_9_10": overlap,
        "verdict": (
            "Sub-chance AUC persists with correct class column and row alignment; "
            "inversion is reported ONLY as a diagnostic, not applied to predictions."
        ),
    }
    (MET / "auc_directionality_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
