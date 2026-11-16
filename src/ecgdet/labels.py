"""Label mapping across datasets to the ELTA 8-class taxonomy.

Target classes (from Papers/Введение в проект.docx): NSR, AF, AFL, SA, SB, ST, SVT, VT
"""

from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "TARGET_CLASSES",
    "map_chapman_dx_to_rhythm",
    "map_ptbxl_scp_to_rhythm",
    "map_mitbih_to_rhythm",
    "map_incart_to_rhythm",
]

TARGET_CLASSES = list(config.RHYTHM_CLASSES)


def map_chapman_dx_to_rhythm(dx_codes: list[str] | str) -> str | None:
    if isinstance(dx_codes, str):
        dx_codes = [c.strip() for c in dx_codes.split(",") if c.strip()]
    for code in dx_codes:
        code_s = str(code).strip()
        if code_s == config.SNOMED_NS_RHYTHM or code_s == '426783006':
            return "NSR"
        if code_s == config.SNOMED_AFIB or code_s == '164889003':
            return "AF"
        if code_s == config.SNOMED_AFL or code_s == '164890007':
            return "AFL"
        if code_s == config.SNOMED_SA or code_s == '427393009':
            return "SA"
        if code_s == config.SNOMED_SB or code_s == '426177001':
            return "SB"
        if code_s == config.SNOMED_ST or code_s == '427084000':
            return "ST"
        if code_s == config.SNOMED_SVT or code_s == '426761007':
            return "SVT"
        # VT / ventricular tachycardia
        if code_s in (getattr(config, 'SNOMED_VT', None), '164895007', '111288001', '25569003', '17338001', '713427006'):
            return "VT"
        c_up = code_s.upper()
        if 'VENTRICULAR TACH' in c_up or c_up == 'VT':
            return "VT"
    return None


def map_ptbxl_scp_to_rhythm(scp_codes) -> str | None:
    if isinstance(scp_codes, float) and pd.isna(scp_codes):
        return None
    if isinstance(scp_codes, str):
        try:
            scp_codes = ast.literal_eval(scp_codes)
        except (ValueError, SyntaxError):
            # sometimes stored differently
            if '{' in scp_codes:
                return None
            # space/comma separated
            return None
    if not isinstance(scp_codes, dict):
        return None
    keys = set(scp_codes.keys())
    if "AFIB" in keys:
        return "AF"
    if "AFLT" in keys:
        return "AFL"
    if "SR" in keys:
        return "NSR"
    if "SARRH" in keys or "SA" in keys:
        return "SA"
    if "SBRAD" in keys:
        return "SB"
    if "STACH" in keys:
        return "ST"
    if "SVTAC" in keys or "PSVT" in keys or "AVNRT" in keys or "AVRT" in keys or "SVARR" in keys:
        return "SVT"
    if "VT" in keys or "VENTACH" in keys or "VFLUT" in keys:
        return "VT"
    return None


def map_mitbih_to_rhythm(diagnosis_comment: str | None) -> str | None:
    # MIT-BIH rhythm annotations are diverse; focus on major rhythm classes if present
    if not diagnosis_comment:
        return None
    s = diagnosis_comment.upper()
    if "ATRIAL FIBRILLATION" in s and "FLUTTER" not in s:
        return "AF"
    if "ATRIAL FLUTTER" in s:
        return "AFL"
    if "VENTRICULAR TACHYCARDIA" in s:
        return "VT"
    if "SUPRAVENTRICULAR TACHYCARDIA" in s:
        return "SVT"
    if "SINUS BRADYCARDIA" in s:
        return "SB"
    if "SINUS TACHYCARDIA" in s:
        return "ST"
    if "SINUS RHYTHM" in s and "BRADY" not in s and "TACHY" not in s:
        return "NSR"
    return None


def map_incart_to_rhythm(diagnosis: str) -> str | None:
    if diagnosis is None:
        return None
    if isinstance(diagnosis, float) and pd.isna(diagnosis):
        return None
    s = str(diagnosis).upper()
    if "ATRIAL FIBRILLATION" in s:
        return "AF"
    if "ATRIAL FLUTTER" in s:
        return "AFL"
    if "VENTRICULAR TACHYCARDIA" in s:
        return "VT"
    if "SUPRAVENTRICULAR" in s and "TACHYCARDIA" in s:
        return "SVT"
    if "SINUS BRADYCARDIA" in s:
        return "SB"
    if "SINUS TACHYCARDIA" in s:
        return "ST"
    if "SINUS RHYTHM" in s:
        return "NSR"
    if "SINUS ARRHYTHMIA" in s:
        return "SA"
    return None
