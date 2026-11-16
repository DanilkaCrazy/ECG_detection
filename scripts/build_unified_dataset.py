"""Build a unified training dataset from multiple sources with patient-wise splits.

Strategy (to avoid leakage):
- Prefer Chapman-Shaoxing for the ELTA target classes (10s, 12-lead) as primary
- Add PTB-XL where codes map cleanly, keeping strat_fold in mind
- MIT-BIH/INCART are long recordings (30min) - extract whole-record or
  windowed features but they may have different patient sets; treat cautiously
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from . import config, labels
from .wfdb_io import parse_header, read_mat_signal, parse_header_lenient

__all__ = ["build_unified_manifest"]


def _load_chapman_manifest() -> pd.DataFrame:
    path = config.REPORTS_ROOT / "manifests" / "chapman_manifest.csv"
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def _load_ptbxl_manifest() -> pd.DataFrame:
    path = config.REPORTS_ROOT / "manifests" / "ptbxl_manifest.csv"
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def _load_mitbih_manifest() -> pd.DataFrame:
    path = config.REPORTS_ROOT / "manifests" / "mitbih_manifest.csv"
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def _load_incart_manifest() -> pd.DataFrame:
    path = config.REPORTS_ROOT / "manifests" / "incart_manifest.csv"
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def build_unified_manifest(output_dir: str | Path | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---------------------------------------------------------------- chapman
    chap_root = config.dataset_root("chapman") / config.CHAPMAN_SUBDIR / config.CHAPMAN_RECORDS_ROOT
    chap_meta = _load_chapman_manifest()
    for _, m in chap_meta.iterrows():
        record = m["record"]
        hea = chap_root / m["folder"] / f"{record}.hea"
        mat = chap_root / m["folder"] / f"{record}.mat"
        try:
            header, note = parse_header_lenient(hea, mat)
        except Exception:
            continue
        if header is None:
            continue
        rhythm = labels.map_chapman_dx_to_rhythm(header.dx_codes)
        if rhythm is None:
            continue
        rows.append(
            {
                "source": "chapman",
                "record_id": record,
                "patient_id": None,
                "rhythm": rhythm,
                "fs": header.fs,
                "n_samp": header.n_samp,
                "leads": ",".join(header.lead_names),
                "path_hea": str(hea),
                "path_mat": str(mat),
                "path_dat": None,
                "duration_s": header.n_samp / header.fs if header.fs and header.n_samp else None,
            }
        )

    # ---------------------------------------------------------------- ptb-xl
    ptb_root = config.dataset_root("ptbxl") / "physionet.org/files/ptb-xl/1.0.3"
    ptb_meta = _load_ptbxl_manifest()
    scp_csv = ptb_root / "scp_statements.csv"
    for _, m in ptb_meta.iterrows():
        scp_codes = m.get("scp_codes")
        rhythm = labels.map_ptbxl_scp_to_rhythm(scp_codes)
        if rhythm is None:
            continue
        filename_hr = m["filename_he"]
        rows.append(
            {
                "source": "ptbxl",
                "record_id": str(m["record"]),
                "patient_id": str(m["patient_id"]) if pd.notna(m.get("patient_id")) else None,
                "rhythm": rhythm,
                "fs": 500.0,
                "n_samp": 5000,
                "leads": "I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6",
                "path_hea": str(ptb_root / f"{filename_hr}.hea"),
                "path_mat": None,
                "path_dat": str(ptb_root / f"{filename_hr}.dat"),
                "duration_s": 10.0,
                "strat_fold": m.get("strat_fold"),
            }
        )

    # ---------------------------------------------------------------- mit-bih
    mit_root = config.dataset_root("mitbih") / "physionet.org/files/mitdb/1.0.0"
    mit_meta = _load_mitbih_manifest()
    for _, m in mit_meta.iterrows():
        rec = m["record"]
        try:
            header = parse_header(mit_root / f"{rec}.hea")
        except Exception:
            continue
        rhythm = labels.map_mitbih_to_rhythm(header.comments.get("Dx"))
        if rhythm is None:
            continue
        rows.append(
            {
                "source": "mitbih",
                "record_id": rec,
                "patient_id": rec,
                "rhythm": rhythm,
                "fs": header.fs,
                "n_samp": header.n_samp,
                "leads": ",".join(header.lead_names),
                "path_hea": str(mit_root / f"{rec}.hea"),
                "path_mat": None,
                "path_dat": str(mit_root / f"{rec}.dat"),
                "duration_s": header.n_samp / header.fs if header.fs else None,
            }
        )

    # ---------------------------------------------------------------- incart
    inc_root = config.dataset_root("incart") / "physionet.org/files/incartdb/1.0.0"
    inc_meta = _load_incart_manifest()
    for _, m in inc_meta.iterrows():
        rec = m["record"]
        try:
            header = parse_header(inc_root / f"{rec}.hea")
        except Exception:
            continue
        rhythm = labels.map_incart_to_rhythm(m.get("diagnosis"))
        if rhythm is None:
            continue
        rows.append(
            {
                "source": "incart",
                "record_id": rec,
                "patient_id": m.get("patient_id") or rec,
                "rhythm": rhythm,
                "fs": header.fs,
                "n_samp": header.n_samp,
                "leads": ",".join(header.lead_names),
                "path_hea": str(inc_root / f"{rec}.hea"),
                "path_mat": None,
                "path_dat": str(inc_root / f"{rec}.dat"),
                "duration_s": header.n_samp / header.fs if header.fs else None,
            }
        )

    df = pd.DataFrame(rows)
    out = output_dir / "unified_ecg_manifest.csv"
    df.to_csv(out, index=False)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_records": int(len(df)),
        "by_source": df.groupby("source").size().to_dict(),
        "by_rhythm": df.groupby("rhythm").size().to_dict(),
    }
    (output_dir / "unified_ecg_manifest_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return df
