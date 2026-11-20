"""Stage 3.6.9 — AFIB/AFL label audit across ECG sources (P0).

The audit is deliberately conservative:

* every source declares explicitly which raw rhythm strings mean AFIB and AFL;
* no string matching on substrings like ``AF`` is allowed;
* a record is ``OK`` only when the observed ``label`` matches the class implied
  by its raw rhythm string in that source;
* a source whose rhythm/label direction is inverted is a hard error, not
  something to repair by flipping predictions at evaluation time.

Known source encodings (verified from the build scripts / raw metadata):

* Chapman: ``#Dx`` SNOMED codes; primary code ``164889003`` -> AF,
  ``164890007`` -> AFL.  In the processed table the rhythm column uses ``AF``.
* PTB-XL: ``scp_codes`` keys ``AFIB`` / ``AFLT``.
* CPSC2018-extra: ``#Dx`` SNOMED ``164889003`` / ``164890007``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CLASS_BY_RHYTHM",
    "SOURCE_RHYTHM_CODES",
    "SNOMED_AFIB",
    "SNOMED_AFL",
    "build_label_table",
    "audit_labels",
    "check_label_direction",
]

SNOMED_AFIB = "164889003"
SNOMED_AFL = "164890007"

# Canonical target label: 1 = AFIB (fibrillation), 0 = AFL (flutter).
CLASS_BY_RHYTHM: dict[str, int] = {"AF": 1, "AFIB": 1, "AFL": 0}

SOURCE_RHYTHM_CODES: dict[str, dict[str, tuple[str, ...]]] = {
    "chapman": {"afib": ("AF",), "afl": ("AFL",)},
    "ptbxl": {"afib": ("AFIB",), "afl": ("AFL",)},
    "cpsc2018": {"afib": ("AFIB",), "afl": ("AFL",)},
    "cpsc2018_extra": {"afib": ("AFIB",), "afl": ("AFL",)},
}

EXCLUDED_CATEGORIES = ("AFIB_AFL_UNCERTAIN", "MULTIPLE_CONFLICTING_RHYTHMS")


def _normalise_source(value: Any) -> str:
    return str(value).strip().lower()


def expected_label(source: Any, rhythm: Any) -> int | None:
    """Return the canonical class implied by source + raw rhythm string."""
    src = _normalise_source(source)
    rules = SOURCE_RHYTHM_CODES.get(src)
    if rules is None:
        return None
    raw = str(rhythm).strip().upper() if not pd.isna(rhythm) else ""
    for code in rules["afib"]:
        if raw == code.upper():
            return CLASS_BY_RHYTHM["AFIB"]
    for code in rules["afl"]:
        if raw == code.upper():
            return CLASS_BY_RHYTHM["AFL"]
    return None


def build_label_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-record table: raw rhythm, expected class, observed class, status."""
    required = {"source", "rhythm", "label"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"meta is missing columns: {sorted(missing)}")

    rows = []
    for i, rec in meta.iterrows():
        exp = expected_label(rec["source"], rec["rhythm"])
        obs = int(rec["label"])
        known_source = _normalise_source(rec["source"]) in SOURCE_RHYTHM_CODES
        if not known_source:
            status, reason = "UNKNOWN_SOURCE", "source has no declared label rules"
        elif exp is None:
            status, reason = "UNKNOWN_CODE", f"rhythm {rec['rhythm']!r} not declared for source"
        elif exp != obs:
            status, reason = "CONFLICT", f"rhythm implies {exp}, meta.label={obs}"
        else:
            status, reason = "OK", "raw rhythm and label agree"
        rows.append({
            "row": i,
            "source": rec["source"],
            "record_id": rec.get("record_id", np.nan),
            "rhythm": rec["rhythm"],
            "expected_label": exp,
            "observed_label": obs,
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def check_label_direction(meta: pd.DataFrame) -> bool:
    """Fail loudly if any source maps a rhythm to the opposite class.

    This is the guard that prevents the PTB-XL ROC-AUC ~0.04 result from being
    'fixed' by silently inverting probabilities.  If direction is wrong the
    data must be repaired, not the metric.
    """
    table = build_label_table(meta)
    unknown_sources = sorted(set(table.loc[table.status == "UNKNOWN_SOURCE", "source"]))
    if unknown_sources:
        raise ValueError(
            f"label direction cannot be verified for undeclared sources: {unknown_sources}"
        )
    # A handful of conflicting records is a data-quality problem, not evidence
    # of an inverted convention.  Only a *systematic* disagreement means the
    # source encodes the two classes in the opposite order.
    for src, sub in table.groupby("source", sort=True):
        known = sub[sub.expected_label.notna()]
        if len(known) == 0:
            continue
        agree = float((known.observed_label == known.expected_label).mean())
        if agree < 0.5:
            raise ValueError(
                f"label direction conflict in source {src!r}: only {agree:.1%} of records "
                "follow the declared AFIB=1 / AFL=0 convention"
            )
    return True


def audit_labels(meta: pd.DataFrame) -> dict[str, Any]:
    """Full label audit: distribution, conflicts, multi-diagnosis, direction."""
    table = build_label_table(meta)
    by_source: dict[str, Any] = {}
    for src, sub in table.groupby("source", sort=True):
        by_source[str(src)] = {
            "n": int(len(sub)),
            "n_afib": int((sub.observed_label == 1).sum()),
            "n_afl": int((sub.observed_label == 0).sum()),
            "n_conflicts": int((sub.status == "CONFLICT").sum()),
            "n_unknown": int((~sub.status.isin(["OK", "CONFLICT"])).sum()),
            "rhythms": Counter(sub.rhythm.astype(str)).most_common(),
        }
    direction_ok = True
    try:
        check_label_direction(meta)
    except ValueError:
        direction_ok = False
    return {
        "n_records": int(len(meta)),
        "n_conflicts": int((table.status == "CONFLICT").sum()),
        "n_unknown": int((~table.status.isin(["OK", "CONFLICT"])).sum()),
        "direction_ok": direction_ok,
        "by_source": by_source,
        "class_distribution": {int(k): int(v) for k, v in meta.label.value_counts().items()},
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "table": table,
    }
