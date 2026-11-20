"""Dataset harmonization contract for multi-source AFIB/AFL ECG data (Stage 3.0 FINAL).

Key rules:
- physical units are checked explicitly; unknown units are NEVER assumed mV;
- WFDB `p_signal` is already in physical units (adc_gain/baseline already applied):
  applying gain a second time is forbidden; for raw digital `d_signal` the
  conversion is (digital - baseline) / adc_gain;
- QC statistics are computed BEFORE NaN/Inf replacement;
- lead mapping is by name and channel indices are validated against the actual
  signal width; missing leads are represented by an explicit mask;
- physically implausible amplitudes are flagged invalid/require_review.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "CANONICAL_LEADS", "ECGRecord", "HarmonizationStatus",
    "canonicalize_leads", "harmonize_signal", "signal_quality", "signal_hash",
    "decimate_signal",
]

CANONICAL_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

_LEAD_ALIASES = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "AR": "aVR", "AL": "aVL", "AF": "aVF",
    "MLII": "II", "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
    # ECG1/ECG2 are unnamed Holter channels: no canonical 12-lead mapping.
}

# Physical ECG amplitudes in mV: typical |amplitude| < 2 mV, std < 1 mV.
# Larger values usually mean mV/uV confusion or gain misapplication.
MAX_ABS_MV = 20.0
MAX_STD_MV = 5.0


class HarmonizationStatus:
    OK = "ok"
    REQUIRE_REVIEW = "require_review"
    INVALID = "invalid"


@dataclass
class ECGRecord:
    record_id: str
    source: str
    patient_id: str | None
    rhythm: str
    diagnosis_codes: list[str] = field(default_factory=list)
    primary_rhythm: str | None = None
    fs: float = 500.0
    signal_units: str = "mV"
    adc_gain: float | None = None
    baseline: float | None = None
    lead_names: list[str] = field(default_factory=list)
    split: str | None = None
    signal: np.ndarray | None = None
    lead_mask: np.ndarray | None = None
    qc: dict[str, Any] = field(default_factory=dict)
    status: str = HarmonizationStatus.REQUIRE_REVIEW
    hash: str | None = None
    original_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "source": self.source,
            "patient_id": self.patient_id, "rhythm": self.rhythm,
            "fs": self.fs, "signal_units": self.signal_units,
            "adc_gain": self.adc_gain, "status": self.status,
            "leads_present": int(self.lead_mask.sum()) if self.lead_mask is not None else None,
            "qc": self.qc, "hash": self.hash, "split": self.split,
            "notes": self.notes,
        }


def canonicalize_leads(lead_names: list[str], n_channels: int | None = None) -> tuple[list[int | None], np.ndarray]:
    """Map named channels to canonical 12-lead order.

    Channel indices are checked against `n_channels` when provided, so a name
    without a corresponding physical channel cannot produce mask=True.
    """
    lookup: dict[str, int] = {}
    for i, name in enumerate(lead_names):
        key = str(name).strip().upper().replace(" ", "")
        canon = _LEAD_ALIASES.get(key)
        if canon is not None and canon not in lookup:
            if n_channels is None or i < n_channels:
                lookup[canon] = i
    idx: list[int | None] = [lookup.get(lead) for lead in CANONICAL_LEADS]
    mask = np.array([i is not None for i in idx], dtype=bool)
    return idx, mask


def signal_quality(sig: np.ndarray) -> dict[str, float]:
    """QC computed on the raw signal BEFORE NaN/Inf replacement."""
    s = np.asarray(sig, dtype=np.float64)
    if s.size == 0:
        return {"std": 0.0, "max_abs": 0.0, "nan_frac": 1.0, "inf_frac": 1.0,
                "flat_leads": 0, "clipped_leads": 0}
    return {
        "std": float(np.nanstd(s)),
        "max_abs": float(np.nanmax(np.abs(s))),
        "nan_frac": float(np.isnan(s).mean()),
        "inf_frac": float(np.isinf(s).mean()),
        "flat_leads": int(np.sum(np.nanstd(s, axis=1) < 1e-6)) if s.ndim == 2 else 0,
        "clipped_leads": int(np.sum(np.nanmax(np.abs(s), axis=1) > MAX_ABS_MV)) if s.ndim == 2 else 0,
    }


def signal_hash(sig: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(sig, dtype=np.float32).tobytes()).hexdigest()


def decimate_signal(x: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Rate reduction WITH antialias filtering (Stage 3.6.15).

    Plain ``x[::k]`` aliases components above the new Nyquist frequency into
    the atrial band.  ``resample_poly`` low-pass filters before decimation.
    Supports 1-D and (leads, samples) arrays along the last axis.
    """
    x = np.asarray(x, dtype=np.float64)
    fs, target_fs = float(fs), float(target_fs)
    if fs <= target_fs:
        return x.astype(np.float32)
    ratio = Fraction(fs / target_fs).limit_denominator(1000)
    return np.asarray(resample_poly(x, ratio.numerator, ratio.denominator, axis=-1),
                      dtype=np.float32)


def harmonize_signal(
    sig: np.ndarray,
    fs: float,
    lead_names: list[str],
    *,
    units: str | list[str],
    signal_kind: str = "physical",      # "physical" (p_signal) or "digital" (d_signal)
    adc_gain: float | np.ndarray | None = None,
    baseline: float | np.ndarray | None = None,
    target_fs: float = 500.0,
    target_len: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Harmonize an arbitrary ECG to canonical (12, target_len) mV signal.

    `units` must be explicit: 'mV', 'uV'/'µV', or 'digital'. Unknown units yield
    status require_review/invalid and no conversion is guessed.
    For WFDB p_signal, pass signal_kind='physical' and do NOT supply gain again.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    info: dict[str, Any] = {
        "original_shape": list(sig.shape), "fs_in": float(fs),
        "units_in": str(units), "signal_kind": signal_kind,
        "lead_names_in": list(lead_names),
        "status": HarmonizationStatus.OK, "notes": [],
    }

    # QC BEFORE cleaning — damaged signals must never be reported OK
    info["qc_raw"] = signal_quality(sig)
    if info["qc_raw"]["nan_frac"] > 0 or info["qc_raw"]["inf_frac"] > 0:
        info["status"] = HarmonizationStatus.REQUIRE_REVIEW
        info["notes"].append(f"raw NaN/Inf: nan={info['qc_raw']['nan_frac']:.3f}, "
                             f"inf={info['qc_raw']['inf_frac']:.3f}")

    # per-channel units: accept a list matching n_channels, or one string for all
    if isinstance(units, (list, tuple, np.ndarray)):
        unit_list = [str(u).strip().lower() for u in units]
        if len(unit_list) != sig.shape[0]:
            info["status"] = HarmonizationStatus.INVALID
            info["notes"].append(f"units count {len(unit_list)} != channels {sig.shape[0]}")
            return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
        if len(set(unit_list)) > 1:
            info["notes"].append("heterogeneous units across channels; converting each")
    else:
        unit_list = [str(units).strip().lower()] * sig.shape[0]
    u = unit_list[0]
    if signal_kind == "digital" or u in ("digital", "counts", "adc"):
        if adc_gain is None:
            info["status"] = HarmonizationStatus.INVALID
            info["notes"].append("digital signal without adc_gain")
            return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
        gain = np.asarray(adc_gain, dtype=float)
        base = 0.0 if baseline is None else np.asarray(baseline, dtype=float)
        # explicit (channels, 1) broadcasting for per-channel gain/baseline
        if gain.ndim == 1:
            if gain.size != sig.shape[0]:
                info["status"] = HarmonizationStatus.INVALID
                info["notes"].append("adc_gain length != channels")
                return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
            gain = gain[:, None]
        if np.ndim(base) == 1:
            base = np.asarray(base, dtype=float)
            if base.size != sig.shape[0]:
                info["status"] = HarmonizationStatus.INVALID
                info["notes"].append("baseline length != channels")
                return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
            base = base[:, None]
        if np.any(np.abs(gain) < 1e-12):
            info["status"] = HarmonizationStatus.INVALID
            info["notes"].append("zero adc_gain")
            return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
        sig = (sig - base) / gain
        info["unit_conversion"] = "digital -> mV via (x-baseline)/adc_gain"
    elif all(x in ("mv", "millivolt") for x in unit_list):
        info["unit_conversion"] = "physical mV (p_signal); gain already applied by WFDB"
    elif all(x in ("uv", "microv", "µv", "microvolt") for x in unit_list):
        sig = sig / 1000.0
        info["unit_conversion"] = "physical uV -> mV"
    elif any(x not in ("mv", "millivolt", "uv", "microv", "µv", "microvolt") for x in unit_list):
        info["status"] = HarmonizationStatus.REQUIRE_REVIEW
        info["notes"].append(f"unknown physical units {sorted(set(unit_list))}; refusing to assume mV")
        info["unit_conversion"] = "NONE (unknown units)"
        return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
    else:
        # mixed mV/uV -> convert uV channels individually
        for ci, un in enumerate(unit_list):
            if un in ("uv", "microv", "µv", "microvolt"):
                sig[ci] /= 1000.0
        info["unit_conversion"] = "per-channel mV/uV -> mV"

    idx, mask = canonicalize_leads(lead_names, n_channels=sig.shape[0])
    info["lead_mask"] = mask.tolist()
    info["missing_leads"] = [lead for lead, ok in zip(CANONICAL_LEADS, mask) if not ok]
    if info["missing_leads"]:
        info["status"] = HarmonizationStatus.REQUIRE_REVIEW
        info["notes"].append(f"missing leads: {info['missing_leads']}")
    out = np.zeros((len(CANONICAL_LEADS), sig.shape[1]), dtype=np.float64)
    for j, ci in enumerate(idx):
        if ci is not None and ci < sig.shape[0]:
            out[j] = sig[ci]
        else:
            info["notes"].append(f"lead {CANONICAL_LEADS[j]} has no valid channel index")

    if abs(float(fs) - target_fs) > 1e-6:
        # rational approximation for non-integer fs (e.g. 257 Hz), bounded denominator
        ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(10000)
        up, down = ratio.numerator, ratio.denominator
        if not np.isclose(up / down, float(target_fs) / float(fs), rtol=1e-4):
            info["status"] = HarmonizationStatus.INVALID
            info["notes"].append(f"cannot safely resample fs={fs}")
            return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
        out = resample_poly(out, up, down, axis=1)
        info["resampled"] = True
        info["resample_ratio"] = f"{up}/{down}"
    else:
        info["resampled"] = False

    if out.shape[1] < target_len:
        info["status"] = HarmonizationStatus.REQUIRE_REVIEW
        info["notes"].append(f"too short: {out.shape[1]} < {target_len}; zero padding is not a real signal")
        out = np.pad(out, ((0, 0), (0, target_len - out.shape[1])), mode="constant")
    else:
        out = out[:, :target_len]

    # physical plausibility check on cleaned signal
    max_abs = float(np.nanmax(np.abs(out)))
    std = float(np.nanstd(out))
    info["qc_clean"] = {"max_abs": max_abs, "std": std}
    if max_abs > MAX_ABS_MV or std > MAX_STD_MV:
        info["status"] = HarmonizationStatus.REQUIRE_REVIEW
        info["notes"].append(f"implausible amplitude after conversion: max={max_abs:.2f} mV, std={std:.2f} mV")

    if out.shape != (len(CANONICAL_LEADS), target_len):
        info["status"] = HarmonizationStatus.INVALID
        info["notes"].append(f"bad output shape {out.shape}")
        return np.zeros((12, target_len), dtype=np.float32), np.zeros(12, dtype=bool), info
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32), mask, info
