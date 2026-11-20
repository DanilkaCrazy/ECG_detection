"""Stage 3.6.14 — compare re-derived labels with the old stage34 labels."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3614"


def main() -> int:
    prov = pd.read_csv(OUT / "diagnosis_provenance.csv")
    old = pd.read_csv(ROOT / "data" / "processed" / "af_afl_stage34" / "meta.csv")
    df = prov.merge(old[["source", "record_id", "label", "rhythm", "split"]],
                    on=["source", "record_id"], how="left", suffixes=("_new", "_old"))
    valid = df.label_status.isin(["AFIB", "AFL"])
    df["agrees_with_old"] = valid & (df.label_new == df.label_old)
    df["old_label_invalid"] = ~valid
    rows = []
    for src, sub in df.groupby("source"):
        rows.append({
            "source": src, "n": len(sub),
            "n_valid": int(valid[sub.index].sum()),
            "n_ambiguous": int((sub.label_status == "AMBIGUOUS").sum()),
            "n_other": int((sub.label_status == "OTHER").sum()),
            "n_unknown": int((sub.label_status == "UNKNOWN").sum()),
            "n_old_label_invalid": int((~valid[sub.index]).sum()),
            "n_valid_but_disagrees": int((valid[sub.index] &
                                          (sub.label_new != sub.label_old)).sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(MET / "label_audit_by_source.csv", index=False)
    df[["source", "record_id", "label_old", "label_new", "label_status",
        "agrees_with_old", "old_label_invalid", "raw_diagnosis_codes",
        "raw_diagnosis_values"]].to_csv(MET / "label_audit_full.csv", index=False)
    summary = {
        "n_records": int(len(df)),
        "n_valid": int(valid.sum()),
        "n_old_labels_without_valid_diagnosis": int((~valid).sum()),
        "n_ambiguous": int((df.label_status == "AMBIGUOUS").sum()),
        "by_source": result.to_dict("records"),
        "conclusion": (
            "Stage 3.6.14 found that a large share of stage34 labels cannot be "
            "re-derived from the original diagnosis metadata. PTB-XL AFIB "
            "records mostly carry AFIB with likelihood 0.0, which the old "
            "builder treated as a positive diagnosis."
        ),
    }
    (MET / "label_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_source"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
