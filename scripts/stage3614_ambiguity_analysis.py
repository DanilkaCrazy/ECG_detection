"""Stage 3.6.14 — analyse ambiguous / mixed / zero-likelihood diagnoses."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3614"


def main() -> int:
    df = pd.read_csv(OUT / "diagnosis_provenance.csv")
    amb = df[df.label_status.isin(["AMBIGUOUS", "OTHER", "UNKNOWN"])].copy()
    amb.to_csv(OUT / "ambiguous_records.csv", index=False)
    rows = []
    for src, sub in df.groupby("source"):
        rows.append({
            "source": src,
            "n": len(sub),
            "n_ambiguous": int((sub.label_status == "AMBIGUOUS").sum()),
            "n_other": int((sub.label_status == "OTHER").sum()),
            "n_unknown": int((sub.label_status == "UNKNOWN").sum()),
            "n_zero_likelihood_target": int(sub.notes.astype(str).str.contains(
                "likelihood 0").sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(MET / "ambiguity_summary.csv", index=False)
    report = {
        "by_source": summary.to_dict("records"),
        "total_ambiguous": int((df.label_status == "AMBIGUOUS").sum()),
        "total_other": int((df.label_status == "OTHER").sum()),
        "total_zero_likelihood_target": int(df.notes.astype(str).str.contains(
            "likelihood 0").sum()),
        "policy": "AMBIGUOUS / OTHER / UNKNOWN records never receive a binary label",
    }
    (MET / "ambiguity_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k != "by_source"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
