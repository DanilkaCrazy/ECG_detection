"""Stage 3.6.14 — ONE authoritative binary label resolver.

The previous codebase had three competing rules ("first recognised code",
"AFIB before AFLT", and a config file saying ambiguous records must be
excluded).  This module replaces them: a record with BOTH AFIB and AFL codes is
always ``AMBIGUOUS`` and never receives a binary label.

Canonical SNOMED CT (verified against the official ConditionNames table):

    164889003 = Atrial fibrillation -> AFIB -> label 1
    164890007 = Atrial flutter      -> AFL  -> label 0
"""
from __future__ import annotations

from typing import Iterable

from .diagnosis_provenance import DiagnosisRecord

__all__ = [
    "MAPPING_VERSION", "CANONICAL_SNOMED", "AFIB_CODES", "AFL_CODES",
    "CHAPMAN_RAW_ACRONYM", "LABEL_STATUSES",
    "resolve_binary", "resolve_diagnosis_codes", "resolve_chapman_acronym",
]

MAPPING_VERSION = "1.0"

CANONICAL_SNOMED: dict[str, tuple[str, int]] = {
    "164889003": ("AFIB", 1),
    "164890007": ("AFL", 0),
}

# ONLY unambiguous source codes live here.  SNOMED 164934002 is "T wave
# Change" (TWC) in the official Chapman ConditionNames table — including it
# in AFIB_CODES was a real bug.  The bare string "AF" is deliberately absent:
# in the RAW Chapman taxonomy it means Atrial Flutter, while the project's
# internal harmonised rhythm column uses "AF" for fibrillation.  The two
# namespaces must never be mixed (see resolve_chapman_acronym).
AFIB_CODES = frozenset({"164889003", "AFIB"})   # SNOMED | SCP
AFL_CODES = frozenset({"164890007", "AFLT"})    # SNOMED | SCP

# Official ConditionNames_SNOMED-CT.csv acronyms (verified 2026-11-20).
CHAPMAN_RAW_ACRONYM: dict[str, tuple[str, str | None]] = {
    "AFIB": ("AFIB", 1),
    "AF": ("AFL", 0),        # raw Chapman acronym = Atrial Flutter
    "AFL": ("AFL", 0),
    "TWC": ("OTHER", None),  # 164934002 T wave Change — not AFIB
}

LABEL_STATUSES = ("AFIB", "AFL", "MIXED", "OTHER", "AMBIGUOUS", "UNKNOWN", "UNREADABLE")


def resolve_chapman_acronym(acronym: str) -> str:
    """Resolve a RAW Chapman acronym (namespace: ConditionNames table)."""
    key = str(acronym).strip().upper()
    return CHAPMAN_RAW_ACRONYM.get(key, ("UNKNOWN", None))[0]


def _classify(codes: Iterable[str]) -> tuple[str, int | None]:
    seq = [str(c).strip() for c in codes]
    has_afib = any(c in AFIB_CODES for c in seq)
    has_afl = any(c in AFL_CODES for c in seq)
    if has_afib and has_afl:
        return "AMBIGUOUS", None
    if has_afib:
        return "AFIB", 1
    if has_afl:
        return "AFL", 0
    if any(c in CANONICAL_SNOMED or c.startswith("164") for c in seq):
        return "OTHER", None
    return ("OTHER", None) if seq else ("UNKNOWN", None)


def resolve_binary(record: DiagnosisRecord) -> DiagnosisRecord:
    """Attach canonical_rhythm / label / label_status in place.

    A code whose likelihood is explicitly ``0`` is a statement that is
    ABSENT, not a diagnosis.  This is critical for PTB-XL: its ``scp_codes``
    dictionary contains zero-valued keys for every non-selected statement, and
    the old builder used ``"AFIB" in codes`` regardless of the value.
    """
    codes = [c for c in record.raw_diagnosis_codes
             if float(record.raw_diagnosis_values.get(c, 1.0) or 0.0) > 0]
    zero_targets = [c for c in record.raw_diagnosis_codes
                    if c in (AFIB_CODES | AFL_CODES)
                    and float(record.raw_diagnosis_values.get(c, 1.0) or 0.0) <= 0]
    if zero_targets:
        record.notes.append(f"target codes with likelihood 0 ignored: {zero_targets}")
    status, label = _classify(codes)
    record.canonical_rhythm = status if status in ("AFIB", "AFL") else None
    record.label = label
    record.label_status = status
    record.mapping_version = MAPPING_VERSION
    return record


def resolve_diagnosis_codes(codes: Iterable[str], source: str = "unknown") -> DiagnosisRecord:
    """Convenience resolver for tests and callers that only have codes."""
    seq = [str(c).strip() for c in codes]
    rec = DiagnosisRecord(
        source=source, record_id="<in-memory>", raw_diagnosis_codes=seq,
        raw_diagnosis_values={}, primary_diagnosis=seq[0] if seq else None,
        rhythm_codes=seq, secondary_diagnoses=seq[1:],
    )
    return resolve_binary(rec)
