"""Paths, taxonomies and constants shared across the ECG project."""

from __future__ import annotations

import os
from pathlib import Path

# Single authoritative mapping version (Stage 3.6.22).  Every module, config
# and dataset manifest must use this value.
LABEL_MAPPING_VERSION = "1.1"

# --------------------------------------------------------------------------- paths


def _detect_project_root() -> Path:
    """Locate the project root (the directory that contains ``data/raw``)."""
    env = os.environ.get("ECG_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in [here.parents[3], here.parents[2], *here.parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate
    return here.parents[3]


PROJECT_ROOT = _detect_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
EXTERNAL_ROOT = DATA_ROOT / "external"
REPORTS_ROOT = PROJECT_ROOT / "reports"
FIGURES_ROOT = REPORTS_ROOT / "figures"

DATASETS = ("chapman", "ptbxl", "mitbih", "incart")


def dataset_root(name: str) -> Path:
    return RAW_ROOT / name


# ------------------------------------------------------------------ Chapman layout

CHAPMAN_SUBDIR = Path("physionet.org/files/ecg-arrhythmia/1.0.0")
CHAPMAN_RECORDS_ROOT = "WFDBRecords"

# ------------------------------------------------------- Chapman SNOMED -> rhythm class
# Source: data/raw/chapman/.../ConditionNames_SNOMED-CT.csv
# Stage 3.6.14: canonical SNOMED CT mapping.  164889003 = atrial
# FIBRILLATION -> "AF" in the ELTA taxonomy; 164890007 = atrial FLUTTER -> "AFL".
# The previous conflicting values ("AFIB"/"AF") are removed; the authoritative
# binary resolver lives in ecgdet.label_resolution.
CHAPMAN_SNOMED_TO_ACRONYM = {
    "426783006": "SR",
    "164889003": "AF",
    "164890007": "AFL",
    "427393009": "SA",
    "426177001": "SB",
    "427084000": "ST",
    "426761007": "SVT",
    "713422000": "AT",
    "233896004": "AVNRT",
    "233897008": "AVRT",
    "195101003": "SAAWR",
    "164895007": "VT",
    "111288001": "VT",
    "25569003": "VT",
    "17338001": "VT",
}

# ELTA target taxonomy (8 rhythm classes, see Papers/�'�?��?��?��� �? ���?�?���'.docx)
RHYTHM_CLASSES = ("NSR", "AF", "AFL", "SA", "SB", "ST", "SVT", "VT")

# SNOMED codes that define the target rhythm classes
SNOMED_NS_RHYTHM = "426783006"
SNOMED_AFIB = "164889003"
SNOMED_AFL = "164890007"
SNOMED_SA = "427393009"
SNOMED_SB = "426177001"
SNOMED_ST = "427084000"
SNOMED_SVT = "426761007"
SNOMED_VT = "164895007"

SNOMED_TO_RHYTHM = {
    SNOMED_NS_RHYTHM: "NSR",
    SNOMED_AFIB: "AF",
    SNOMED_AFL: "AFL",
    SNOMED_SA: "SA",
    SNOMED_SB: "SB",
    SNOMED_ST: "ST",
    SNOMED_SVT: "SVT",
    SNOMED_VT: "VT",
    "111288001": "VT",
    "25569003": "VT",
    "17338001": "VT",
}

# ELTA target taxonomy (8 rhythm classes, see Papers/Введение в проект.docx)
RHYTHM_CLASSES = ("NSR", "AF", "AFL", "SA", "SB", "ST", "SVT", "VT")

# SNOMED codes that define the target rhythm classes
SNOMED_NS_RHYTHM = "426783006"
SNOMED_AFIB = "164889003"
SNOMED_AFL = "164890007"
SNOMED_SA = "427393009"
SNOMED_SB = "426177001"
SNOMED_ST = "427084000"
SNOMED_SVT = "426761007"

SNOMED_TO_RHYTHM = {
    SNOMED_NS_RHYTHM: "NSR",
    SNOMED_AFIB: "AF",
    SNOMED_AFL: "AFL",
    SNOMED_SA: "SA",
    SNOMED_SB: "SB",
    SNOMED_ST: "ST",
    SNOMED_SVT: "SVT",
}

# --------------------------------------------------------------------- MIT-BIH beats
MITBIH_BEAT_SYMBOLS = {
    "N": "N",
    "L": "L",
    "R": "R",
    "A": "A",
    "a": "A",
    "J": "J",
    "S": "S",
    "V": "V",
    "F": "F",
    "e": "e",
    "j": "j",
    "E": "E",
    "/": "V",
    "f": "F",
    "Q": "Q",
    "?": "?",
}

# MIT-BIH records that are commonly excluded (paced beats / unusable signal)
MITBIH_EXCLUDE = {"102", "104", "107"}

MITBIH_LEADS = ("I", "II", "V1", "V2", "V5")

# ------------------------------------------------------------------ INCART metadata
# files-patients-diagnoses.txt: 1-based patient id -> record name / diagnosis
INCART_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "VF")

# ----------------------------------------------------------------- PTB-XL metadata
PTBXL_FILENAME = "ptbxl_database.csv"
PTBXL_SCP_FILENAME = "scp_statements.csv"

# PTB-XL SCP codes used for the target taxonomy
PTBXL_SCP_RHYTHM = {
    "I48": "VT",     # ventricular tachycardia
    "I49": "AFL",    # atrial flutter
    "I47": "AF",     # atrial fibrillation
    "I49.1": "AF",
    "I95": "ST",     # sinus tachycardia (ST deviation is I46 / 'ST' sinus tach?)
    "I49.2": "SVT",
    "I49.3": "AT",
    "I00": "AVNRT",
    "I01": "AVRT",
}

# Human readable names for the diagnosis fields we parse out of headers
CHAPMAN_HEADER_FIELDS = ("Age", "Sex", "Dx", "Rx", "Hx", "Sx")
