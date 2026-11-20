"""Stage 3.6.15 — source classification: metadata, raw ECG, class-conditional."""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.source_classifier import build_source_features, evaluate_source_classifier  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3615"


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    n = len(meta)
    step = max(1, n // 3000)
    idx = np.arange(0, n, step)
    sig = np.asarray(X[idx], dtype=np.float32)
    sources = meta.source.to_numpy()[idx]

    # A: metadata-only (signal-derived technical parameters, no ids/paths/labels)
    feats_a = np.column_stack([
        np.sqrt(np.mean(sig ** 2, axis=(1, 2))),
        np.std(sig, axis=(1, 2)),
        np.percentile(np.abs(sig), 99, axis=(1, 2)),
        np.mean(np.abs(sig) < 1e-8, axis=(1, 2)),
        np.full(len(idx), 500.0),
    ])
    # B: full raw-signal descriptors
    feats_b = build_source_features(sig, 500.0)

    results = {
        "A_metadata": evaluate_source_classifier(feats_a, sources, seed=42),
        "B_raw_ecg": evaluate_source_classifier(feats_b, sources, seed=42),
    }
    # C: class-conditional source recognition
    cond = {}
    for lab in (1, 0):
        sub = np.where(meta.label.to_numpy()[idx] == lab)[0]
        if len(np.unique(sources[sub])) < 2:
            continue
        cond[f"label_{lab}"] = evaluate_source_classifier(
            feats_b[sub], sources[sub], seed=42)
    results["C_class_conditional"] = cond
    results["interpretation"] = (
        "High source accuracy at fixed class means sources differ beyond class prevalence."
    )
    (MET / "source_classifier.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in results.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(k, "acc", round(v["accuracy"], 3), "bal", round(v["balanced_accuracy"], 3),
                  "auc", round(v["roc_auc_ovr"], 3))
    for k, v in results.get("C_class_conditional", {}).items():
        print(k, "acc", round(v["accuracy"], 3), "bal", round(v["balanced_accuracy"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
