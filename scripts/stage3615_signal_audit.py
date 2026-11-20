"""Stage 3.6.15 — physical signal audit on the provenance-clean dataset."""
from __future__ import annotations

import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3615"
MET.mkdir(parents=True, exist_ok=True)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    rows = []
    for i in range(len(meta)):
        sig = np.asarray(X[i], dtype=np.float32)
        rows.append({
            "source": meta.source.iloc[i], "label": int(meta.label.iloc[i]),
            "rms": float(np.sqrt(np.mean(sig ** 2))),
            "mean_abs": float(np.mean(np.abs(sig))),
            "std": float(np.std(sig)),
            "p99_abs": float(np.percentile(np.abs(sig), 99)),
            "zero_frac": float(np.mean(np.abs(sig) < 1e-8)),
            "n_leads": 12,
        })
    df = pd.DataFrame(rows)
    df.to_csv(MET / "signal_quality_by_record.csv", index=False)
    summary = df.groupby("source").agg(
        n=("rms", "size"), rms_mean=("rms", "mean"), rms_std=("rms", "std"),
        p99_mean=("p99_abs", "mean"), zero_frac_mean=("zero_frac", "mean")).reset_index()
    # Class-conditional table: source-specific amplitude vs class signal.
    cond = df.groupby(["source", "label"])[["rms", "p99_abs", "zero_frac"]].median().reset_index()
    summary.to_csv(MET / "signal_audit_by_source.csv", index=False)
    cond.to_csv(MET / "signal_audit_class_conditional.csv", index=False)
    report = {
        "n_records": int(len(df)), "by_source": summary.to_dict("records"),
        "class_conditional": cond.to_dict("records"),
        "caveat": "Amplitude differences may reflect acquisition hardware, not physiology.",
    }
    (MET / "signal_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(cond.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
