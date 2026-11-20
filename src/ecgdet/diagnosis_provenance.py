"""Stage 3.6.14 — lossless extraction of ORIGINAL diagnosis codes.

This module reads source metadata (WFDB ``#Dx`` headers and PTB-XL SCP
statements) rather than trusting the already-normalised ``meta.csv``.  Every
code, its order and its likelihood are preserved so that label conversion can
be audited independently.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from . import config

__all__ = [
    "DiagnosisRecord", "extract_chapman_diagnoses", "extract_ptbxl_diagnoses",
    "extract_cpsc_diagnoses", "to_dataframe", "SCP_RHYTHM_CODES",
]

# SCP-ECG rhythm statements (PTB-XL).  Morphological statements are NOT rhythm.
SCP_RHYTHM_CODES = frozenset({
    "AFIB", "AFLT", "SR", "SARRH", "SBRAD", "STACH", "SVTAC", "PSVT",
    "AVNRT", "AVRT", "SVARR", "VT", "VENTACH", "VFLUT", "AFLT", "NSR",
})
_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "MLII": "II"}


@dataclass
class DiagnosisRecord:
    """Complete provenance of a single record's original diagnosis."""
    source: str
    record_id: str
    raw_diagnosis_codes: list[str]
    raw_diagnosis_values: dict[str, float]
    primary_diagnosis: str | None
    rhythm_codes: list[str]
    secondary_diagnoses: list[str]
    canonical_rhythm: str | None = None
    label: int | None = None
    label_status: str = "UNKNOWN"
    mapping_version: str = "1.0"
    dataset_version: str | None = None
    patient_id: str | None = None
    strat_fold: int | None = None
    notes: list[str] = field(default_factory=list)


def _split_dx(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[,;\s]+", str(text).strip()) if c.strip()]


def _read_dx_line(path: pathlib.Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        m = re.match(r"^#\s*Dx\s*:\s*(.*)$", line.strip(), flags=re.I)
        if m:
            return _split_dx(m.group(1))
    return []


def extract_chapman_diagnoses(header: str | pathlib.Path,
                              record_id: str | None = None) -> DiagnosisRecord:
    """Read ALL ``#Dx`` codes from a Chapman WFDB header, preserving order."""
    p = pathlib.Path(header)
    codes: list[str] = []
    notes: list[str] = []
    try:
        from .wfdb_io import parse_header
        codes = list(parse_header(p).dx_codes)
    except Exception as exc:  # pragma: no cover - fallback for unusual headers
        notes.append(f"parse_header failed: {type(exc).__name__}")
        codes = _read_dx_line(p)
    rhythm = [c for c in codes if c in _snomed_rhythm_codes()]
    return DiagnosisRecord(
        source="chapman", record_id=record_id or p.stem,
        raw_diagnosis_codes=codes, raw_diagnosis_values={c: 1.0 for c in codes},
        primary_diagnosis=codes[0] if codes else None,
        rhythm_codes=rhythm, secondary_diagnoses=codes[1:],
        dataset_version="physionet-ecg-arrhythmia-1.0.0", notes=notes,
    )


def extract_cpsc_diagnoses(header: str | pathlib.Path,
                           record_id: str | None = None) -> DiagnosisRecord:
    """Read ALL SNOMED codes from a CPSC2018-extra ``# Dx`` header."""
    p = pathlib.Path(header)
    codes = _read_dx_line(p)
    rhythm = [c for c in codes if c in _snomed_rhythm_codes()]
    return DiagnosisRecord(
        source="cpsc2018_extra", record_id=record_id or p.stem,
        raw_diagnosis_codes=codes, raw_diagnosis_values={c: 1.0 for c in codes},
        primary_diagnosis=codes[0] if codes else None,
        rhythm_codes=rhythm, secondary_diagnoses=codes[1:],
        dataset_version="cpsc2018-extra",
    )


def _snomed_rhythm_codes() -> frozenset[str]:
    return frozenset(config.SNOMED_TO_RHYTHM.keys()) | {
        "164889003", "164890007", "426783006",
    }


def extract_ptbxl_diagnoses(scp_codes: Any, record_id: str,
                            patient_id: str | None = None,
                            strat_fold: int | None = None) -> DiagnosisRecord:
    """Parse a PTB-XL ``scp_codes`` value WITHOUT collapsing multiple diagnoses."""
    if isinstance(scp_codes, str):
        try:
            scp_codes = ast.literal_eval(scp_codes)
        except (ValueError, SyntaxError):
            scp_codes = {}
    if not isinstance(scp_codes, dict):
        scp_codes = {}
    values = {str(k): float(v) for k, v in scp_codes.items()}
    rhythm = [str(k) for k in scp_codes if str(k) in SCP_RHYTHM_CODES]
    secondary = [str(k) for k in scp_codes if str(k) not in rhythm]
    # The SCP statement category (diagnostic / rhythm / morphology) is kept for
    # auditing; a key's presence does NOT by itself prove a rhythm diagnosis.
    categories = scp_categories()
    notes = [f"categories: { {k: categories.get(k) for k in scp_codes} }"]
    return DiagnosisRecord(
        source="ptbxl", record_id=str(record_id),
        raw_diagnosis_codes=list(values), raw_diagnosis_values=values,
        primary_diagnosis=rhythm[0] if rhythm else None,
        rhythm_codes=rhythm, secondary_diagnoses=secondary,
        patient_id=None if patient_id is None else str(patient_id),
        strat_fold=strat_fold, dataset_version="ptb-xl-1.0.3", notes=notes,
    )


def scp_categories() -> dict[str, str]:
    """Load SCP statement categories from the official PTB-XL table."""
    p = (config.RAW_ROOT / "ptbxl" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
         / "scp_statements.csv")
    if not p.exists():
        return {}
    df = pd.read_csv(p, index_col=0)
    out = {}
    for code, row in df.iterrows():
        if float(row.get("rhythm", 0) or 0) == 1.0:
            out[str(code)] = "rhythm"
        elif float(row.get("diagnostic", 0) or 0) == 1.0:
            out[str(code)] = "diagnostic"
        elif float(row.get("form", 0) or 0) == 1.0:
            out[str(code)] = "form"
        else:
            out[str(code)] = "other"
    return out


def to_dataframe(records: list[DiagnosisRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = asdict(r)
        for key in ("raw_diagnosis_codes", "rhythm_codes", "secondary_diagnoses",
                    "notes"):
            d[key] = json.dumps(d[key], ensure_ascii=False)
        d["raw_diagnosis_values"] = json.dumps(d["raw_diagnosis_values"], ensure_ascii=False)
        rows.append(d)
    return pd.DataFrame(rows)
