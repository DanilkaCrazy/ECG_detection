"""Stage 3.6.14 commit 3 — re-read ORIGINAL diagnosis codes for every record.

Chapman / CPSC: WFDB ``#Dx`` headers.  PTB-XL: ``scp_codes`` +
``scp_statements.csv`` categories.  Nothing here trusts ``meta.rhythm``.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.diagnosis_provenance import (  # noqa: E402
    extract_chapman_diagnoses, extract_cpsc_diagnoses, extract_ptbxl_diagnoses,
    to_dataframe,
)
from ecgdet.label_resolution import resolve_binary  # noqa: E402
from ecgdet import config  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3614"
OUT.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)


_CPSC_HEADERS = {p.stem: p for p in (
    ROOT / "data" / "raw" / "cpsc2018_extra" / "data").rglob("*.hea")}


def cpsc_header(record_id: str) -> pathlib.Path | None:
    # Files live in nested group folders (g1/, g2/, ...); index by stem once.
    return _CPSC_HEADERS.get(record_id)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    ptb = pd.read_csv(config.RAW_ROOT / "ptbxl" / "physionet.org" / "files" /
                      "ptb-xl" / "1.0.3" / "ptbxl_database.csv")
    # CRITICAL: meta.record_id for PTB-XL is the 0-BASED ROW INDEX used by
    # build_external_af_dataset.py (r.name), NOT the ecg_id column (1-based).
    records = []
    missing = 0
    for i, row in meta.iterrows():
        src, rid = row.source, str(row.record_id)
        if src == "chapman":
            p = pathlib.Path(row.orig_path)
            if not p.exists():
                missing += 1
                continue
            rec = extract_chapman_diagnoses(p, rid)
        elif src == "cpsc2018_extra":
            p = cpsc_header(rid)
            if p is None:
                missing += 1
                continue
            rec = extract_cpsc_diagnoses(p, rid)
        elif src == "ptbxl":
            try:
                key = int(rid)
            except ValueError:
                missing += 1
                continue
            if not (0 <= key < len(ptb)):
                missing += 1
                continue
            dbrow = ptb.iloc[key]
            rec = extract_ptbxl_diagnoses(
                dbrow.get("scp_codes"), rid,
                patient_id=dbrow.get("patient_id"), strat_fold=dbrow.get("strat_fold"))
        else:
            missing += 1
            continue
        resolve_binary(rec)
        rec.notes.append(f"original_rhythm={row.rhythm}; original_label={row.label}")
        records.append(rec)

    df = to_dataframe(records)
    # The row mapping is needed for the dataset rebuild; align explicitly.
    keep_idx = []
    for i, row in meta.iterrows():
        src, rid = row.source, str(row.record_id)
        if src == "chapman":
            ok = pathlib.Path(row.orig_path).exists()
        elif src == "cpsc2018_extra":
            ok = cpsc_header(rid) is not None
        else:
            try:
                key = int(rid)
                ok = 0 <= key < len(ptb)
            except ValueError:
                ok = False
        if ok:
            keep_idx.append(i)
    df["row_in_stage34"] = keep_idx[:len(df)]
    df.to_csv(OUT / "diagnosis_provenance.csv", index=False)

    stats = {
        "n_source_records": int(len(meta)),
        "n_extracted": int(len(df)),
        "n_missing_source_files": int(missing),
        "label_status_counts": df.label_status.value_counts().to_dict(),
        "by_source": {f"{s}|{st}": int(n) for (s, st), n in
                      df.groupby("source").label_status.value_counts().items()},
        "records_with_both_codes": int(df.raw_diagnosis_codes.str.contains(
            "164889003.*164890007|164890007.*164889003|AFIB.*AFLT|AFLT.*AFIB",
            regex=True).sum()),
    }
    (MET / "raw_diagnosis_audit.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
