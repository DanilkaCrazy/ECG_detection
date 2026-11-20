"""Stage 3.6.9 (P0) — CORRECTED external test protocol.

Fixes both disqualifying bugs of ``stage36_external_test.py``:
  * the model is fitted on Chapman TRAIN only (never on its internal test);
  * the threshold is selected on Chapman VAL only; external labels are used
    solely for the final metric computation, never for selection.

The in-domain Chapman test split is reported as a control, and probability
direction is asserted before any ROC-AUC is computed.
"""
from __future__ import annotations

import json
import sys
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.external_evaluation import ExternalTestProtocol  # noqa: E402
from ecgdet.label_audit import check_label_direction  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "reports" / "metrics" / "stage369"
OUT.mkdir(parents=True, exist_ok=True)


def prep(X, fit_idx):
    """Robust scaling fitted ONLY on the source training rows."""
    X = np.asarray(X, dtype=np.float64)
    keep = ~np.isnan(X[fit_idx]).all(axis=0)
    X = X[:, keep]
    med = np.nan_to_num(np.nanmedian(X[fit_idx], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit_idx], [25, 75], axis=0)
    scale = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    center = np.nanmedian(X[fit_idx], axis=0)
    return np.clip((X - center) / scale, -20, 20).astype(np.float32)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    check_label_direction(meta)
    F = np.load(D / "features_advanced_expert.npy", mmap_mode="r")
    fit = np.where((meta.source == "chapman") & (meta.split == "train"))[0]
    Xp = prep(F, fit)

    model = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )
    protocol = ExternalTestProtocol(meta, Xp, model)
    result = protocol.run(source="chapman", external_sources=["ptbxl", "cpsc2018_extra"])

    # Within-source reference: if PTB-XL-ONLY training yields AUC > 0.5 on its
    # own test split, the sub-chance cross-domain AUC is a domain shift and not
    # a label inversion (label direction was already checked explicitly above).
    in_source = {}
    for src in ["ptbxl", "cpsc2018_extra"]:
        fit_src = np.where((meta.source == src) & (meta.split == "train"))[0]
        if len(fit_src) < 20 or meta.loc[fit_src, "label"].nunique() < 2:
            in_source[src] = {"status": "insufficient_train_rows"}
            continue
        Xs = prep(F, fit_src)
        mdl = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )
        r = ExternalTestProtocol(meta, Xs, mdl).run(source=src, external_sources=[])
        c = r["control"]
        in_source[src] = {
            "n_test": c["n"] if c else 0,
            "threshold": r["threshold"],
            "roc_auc": c["roc_auc"] if c else None,
            "f1_macro": c["f1_macro"] if c else None,
            "status": "ok",
        }
    result["in_source_reference"] = in_source

    # Drop raw index lists from the persisted JSON (kept in predictions.csv).
    out = {k: v for k, v in result.items()
           if k not in ("train_indices", "val_indices", "source_test_indices")}
    for d in ("control", *result["external"].keys()):
        target = out["control"] if d == "control" else out["external"][d]
        if target:
            target.pop("indices", None)
    (OUT / "external_test_corrected.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for name, block in [("chapman_control", result["control"]),
                        *[(k, v) for k, v in result["external"].items()]]:
        if block:
            rows.append({"evaluation": name, **{k: v for k, v in block.items()
                                                if k != "confusion_matrix"}})
    pd.DataFrame(rows).to_csv(OUT / "external_test_corrected.csv", index=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
