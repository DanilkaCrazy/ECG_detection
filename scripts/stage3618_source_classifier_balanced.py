"""Stage 3.6.18 — balanced source classification with per-source metrics.

Replaces the misleading overall accuracy of Stage 3.6.15.  ``signal_summary``
is the former "A_metadata" experiment (it uses RMS/std/p99/zero-fraction, not
true device metadata).
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.source_classifier import build_source_features  # noqa: E402
from ecgdet.signal_normalization import normalize_per_lead_time  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3616"
MET = ROOT / "reports" / "metrics" / "stage3618"


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D.parent / "af_afl_stage3614" / "X.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    groups = []
    for s in meta.source.unique():
        idx = np.where(meta.source == s)[0]
        groups.extend(rng.choice(idx, size=min(89, len(idx)), replace=False).tolist())
    groups = np.sort(np.asarray(groups, dtype=int))
    sub = meta.iloc[groups].reset_index(drop=True)
    sig = np.asarray(X[groups], dtype=np.float32)
    y = sub.source.to_numpy()
    g = sub.patient_key.to_numpy()
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr, te = next(split.split(sig, y, g))

    results = {}
    for name, data in (("signal_summary", sig), ("raw_ecg_descriptors", sig),
                       ("raw_ecg_normalized", normalize_per_lead_time(sig))):
        if name == "signal_summary":
            f = np.column_stack([
                np.sqrt(np.mean(data ** 2, axis=(1, 2))),
                np.std(data, axis=(1, 2)),
                np.percentile(np.abs(data), 99, axis=(1, 2)),
                np.mean(np.abs(data) < 1e-8, axis=(1, 2)),
            ]).astype(np.float32)
        else:
            f = build_source_features(data, 500.0)
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42).fit(f[tr], y[tr])
        pred = m.predict(f[te])
        rep = classification_report(y[te], pred, output_dict=True, zero_division=0)
        results[name] = {
            "n_train": len(tr), "n_test": len(te),
            "accuracy": float(accuracy_score(y[te], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "per_source": {s: {"recall": rep[s]["recall"], "precision": rep[s]["precision"],
                               "f1": rep[s]["f1-score"]}
                           for s in np.unique(y) if s in rep},
            "confusion_matrix": confusion_matrix(y[te], pred, labels=np.unique(y)).tolist(),
            "labels": np.unique(y).tolist(),
        }
        if name == "raw_ecg_descriptors":
            try:
                import shap
                ex = shap.TreeExplainer(m)
                sv = ex.shap_values(f[te])
                arr = np.asarray(sv)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                results[name]["top_features"] = sorted(
                    zip([f"f{i}" for i in range(f.shape[1])], np.abs(arr).mean(axis=0).tolist()),
                    key=lambda x: -x[1])[:10]
            except Exception:
                results[name]["top_features"] = []

    (MET / "source_classifier_balanced.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in results.items():
        print(k, "acc", round(v["accuracy"], 3), "bal", round(v["balanced_accuracy"], 3),
              "macroF1", round(v["macro_f1"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
