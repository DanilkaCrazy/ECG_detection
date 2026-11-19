"""Dataset harmonization contract for multi-source AFIB/AFL ECG data.

Design goals (Stage 3.0):
- preserve provenance of every signal (source, patient, diagnosis codes);
- map each dataset's physical channels to a canonical 12-lead order BY NAME,
  never by position;
- keep units/gain/fs explicit and convert to one physical representation (mV);
- represent missing leads with an explicit availability mask instead of silently
  filling them with zeros;
- compute QC metrics and a stable signal hash for duplicate detection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "CANONICAL_LEADS",
    "ECGRecord",
    "canonicalize_leads",
    "harmonize_signal",
    "signal_quality",
    "signal_hash",
]

CANONICAL_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Case-insensitive aliases used by Chapman / PTB-XL / CPSC / INCART.
_LEAD_ALIASES = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "AR": "aVR", "AL": "aVL", "AF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    "ECG1": None, "ECG2": None,  # 2-lead Holter: no canonical 12-lead mapping
}


@dataclass
class ECGRecord:
    """Harmonized ECG record with explicit provenance and lead availability."""

    record_id: str
    source: str
    patient_id: str | None
    rhythm: str
    diagnosis_codes: list[str] = field(default_factory=list)
    primary_rhythm: str | None = None
    fs: float = 500.0
    signal_units: str = "mV"
    adc_gain: float | None = None
    lead_names: list[str] = field(default_factory=list)
    split: str | None = None
    signal: np.ndarray | None = None          # (12, n_samples), mV
    lead_mask: np.ndarray | None = None       # (12,) bool — True if real lead present
    qc: dict[str, Any] = field(default_factory=dict)
    hash: str | None = None
    original_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "source": self.source,
            "patient_id": self.patient_id, "rhythm": self.rhythm,
            "fs": self.fs, "signal_units": self.signal_units,
            "leads_present": int(self.lead_mask.sum()) if self.lead_mask is not None else None,
            "signal_shape": list(self.signal.shape) if self.signal is not None else None,
            "qc": self.qc, "hash": self.hash, "split": self.split,
        }


def canonicalize_leads(lead_names: list[str]) -> tuple[list[int | None], np.ndarray]:
    """Return channel indices (one per canonical lead) and an availability mask.

    Missing leads get index None and mask False; the caller MUST keep the mask
    so downstream models know that the channel is absent.
    """
    lookup: dict[str, int] = {}
    for i, name in enumerate(lead_names):
        key = str(name).strip().upper().replace(" ", "")
        canon = _LEAD_ALIASES.get(key)
        # handle forms like "ECG1", "MLII", "V1"
        if canon is None and key in ("MLII", "II"):
            canon = "II"
        if canon is None and key in ("V5",):
            canon = "V5"
        if canon is not None and canon not in lookup:
            lookup[canon] = i
    idx: list[int | None] = [lookup.get(lead) for lead in CANONICAL_LEADS]
    mask = np.array([i is not None for i in idx], dtype=bool)
    return idx, mask


def signal_quality(sig: np.ndarray) -> dict[str, float]:
    """Basic QC on a (n_leads, n_samples) signal in mV."""
    s = np.asarray(sig, dtype=np.float64)
    return {
        "std_uV_equiv": float(np.std(s)) if s.size else 0.0,
        "max_abs": float(np.max(np.abs(s))) if s.size else 0.0,
        "nan_frac": float(np.isnan(s).mean()) if s.size else 1.0,
        "inf_frac": float(np.isinf(s).mean()) if s.size else 1.0,
        "flat_leads": int(np.sum(np.std(s, axis=1) < 1e-6)) if s.ndim == 2 else 0,
        "clipped_leads": int(np.sum(np.max(np.abs(s), axis=1) > 100.0)) if s.ndim == 2 else 0,
    }


def signal_hash(sig: np.ndarray) -> str:
    """Stable hash of a float32 signal (independent of OS/endianness)."""
    return hashlib.md5(np.ascontiguousarray(sig, dtype=np.float32).tobytes()).hexdigest()


def harmonize_signal(
    sig: np.ndarray,
    fs: float,
    lead_names: list[str],
    *,
    target_fs: float = 500.0,
    target_len: int = 5000,
    units: str = "mV",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map arbitrary 12-lead-ish ECG to canonical (12, target_len) mV signal.

    Returns (signal, mask, info). Missing leads are zero-filled in `signal`,
    but `mask` explicitly marks them; callers must not interpret a zero channel
    as a real recording.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    info: dict[str, Any] = {"original_shape": list(sig.shape), "fs_in": float(fs),
                            "units_in": units, "lead_names_in": list(lead_names)}

    # physical units → mV
    u = str(units).lower()
    if u in ("uv", "microv", "µv"):
        sig = sig / 1000.0
        info["unit_conversion"] = "uV -> mV"
    elif u in ("mv", "millivolt"):
        info["unit_conversion"] = "already mV"
    else:
        info["unit_conversion"] = f"unknown ({units}); assumed mV"

    idx, mask = canonicalize_leads(lead_names)
    info["lead_mask"] = mask.tolist()
    info["missing_leads"] = [lead for lead, ok in zip(CANONICAL_LEADS, mask) if not ok]
    out = np.zeros((len(CANONICAL_LEADS), sig.shape[1]), dtype=np.float64)
    for j, ci in enumerate(idx):
        if ci is not None and ci < sig.shape[0]:
            out[j] = sig[ci]

    if abs(float(fs) - target_fs) > 1e-6:
        g = int(np.gcd(int(target_fs), int(fs)))
        out = resample_poly(out, int(target_fs) // g, int(fs) // g, axis=1)
        info["resampled"] = True
    else:
        info["resampled"] = False

    if out.shape[1] < target_len:
        out = np.pad(out, ((0, 0), (0, target_len - out.shape[1])), mode="constant")
    else:
        out = out[:, :target_len]

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    info.update(signal_quality(out))
    return out.astype(np.float32), mask, info
