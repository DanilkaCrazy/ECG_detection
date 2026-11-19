"""Stage 3.6 P0 — correct atrial signal extraction.

 fixes the Stage 3.5 defects found in the audit:
  * explicit (leads, samples) orientation with an automatic transpose guard;
  * QRST cancellation returns the residual signal itself, not only metrics;
  * atrial spectral / temporal analysis consumes that residual;
  * QRS peaks are never used as atrial cycle peaks;
  * preservation is measured INSIDE QRS windows as well as outside;
  * unreliable extractions are marked unavailable (NaN), not zero-filled.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

__all__ = ["extract_atrial_residual"]

_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "MLII": "II"}


def _lead_index(lead_names):
    out = {}
    for i, n in enumerate(lead_names):
        key = str(n).strip().upper()
        key = _ALIASES.get(key, key)
        out.setdefault(key, i)
    return out


def _bandpass(x, fs, lo, hi):
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    if hi >= nyq or lo <= 0 or len(x) < 8:
        return x.copy()
    b, a = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, x)


def _beat_template(x, fs, peaks, method, pre=0.08, post=0.12):
    L1, L2 = int(pre * fs), int(post * fs)
    L = L1 + L2
    beats = [x[p - L1:p + L2] for p in peaks if p - L1 >= 0 and p + L2 <= len(x)]
    if len(beats) < 2:
        return None, np.zeros(len(x), dtype=bool), 0
    B = np.asarray(beats)
    if method == "adaptive":
        corr = np.corrcoef(B)
        ref = int(np.argmax(np.sum(np.nan_to_num(corr), axis=1)))
        good = corr[ref] > 0.6
        template = np.median(B[good], axis=0) if good.sum() >= 2 else np.median(B, axis=0)
    elif method == "beat_aligned":
        template = np.mean(B, axis=0)
    else:  # median
        template = np.median(B, axis=0)
    return template, L1, L2, len(beats)


def _subtract_template(x, fs, peaks, template, L1, L2):
    L = L1 + L2
    w = np.hanning(L)
    out = x.copy()
    mask = np.zeros(len(x), dtype=bool)
    for p in peaks:
        lo, hi = p - L1, p + L2
        if lo >= 0 and hi <= len(x):
            out[lo:hi] -= template * w
            mask[lo:hi] = True
    return out, mask


def _tq_residual(x, fs, peaks, pre_q=0.10, post_t=0.25):
    """Independent baseline: concatenate TQ segments between T end and next QRS."""
    n = len(x)
    peak_list = np.sort(np.asarray(peaks, dtype=int))
    out = np.full(n, np.nan)
    mask = np.zeros(n, dtype=bool)
    start = int(peak_list[0] + post_t * fs) if len(peak_list) else 0
    for p in peak_list[1:]:
        stop = max(start + 8, int(p - pre_q * fs))
        if stop <= n and stop > start:
            out[start:stop] = x[start:stop]
            mask[start:stop] = True
        start = int(p + post_t * fs)
    if len(peak_list):
        stop = min(n, int(peak_list[-1] + (1.0 / 0.8 - post_t) * fs))
        if stop > start:
            out[start:stop] = x[start:stop]
            mask[start:stop] = True
    return out, mask


def _qrs_window_mask(n, fs, peaks, pre=0.08, post=0.12):
    """QRS windows used for template subtraction (also for TQ exclusion)."""
    mask = np.zeros(n, dtype=bool)
    L1, L2 = int(pre * fs), int(post * fs)
    for p in peaks:
        lo, hi = max(0, p - L1), min(n, p + L2)
        mask[lo:hi] = True
    return mask


def _mask_segments(mask):
    """Lengths (samples) of contiguous True runs in a boolean mask."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return (ends - starts) if len(starts) else np.array([], dtype=int)


def extract_atrial_residual(signal, fs, lead_names, method="median", qrs_peaks=None):
    """Extract the atrial component from a multilead ECG.

    Parameters
    ----------
    signal : (n_leads, n_samples) array
        Expected orientation is (leads, samples). A transposed (samples, leads)
        array is detected and transposed automatically; the fact is recorded in
        ``notes``. This guards against the Stage 3.5 transpose bug.
    method : {"median", "adaptive", "beat_aligned", "tq_segment"}
    qrs_peaks : optional QRS peak indices in samples. When omitted, no QRS
        detection is attempted here — pass peaks from ``multi_lead_qrs``.

    Returns
    -------
    dict with residual, qrs_exclusion_mask (QRS windows for ALL methods),
    valid_sample_mask (samples usable for analysis — whole signal for template
    subtraction, TQ segments for tq_segment), qrs_suppression_quality (NaN for
    TQ), residual_energy_ratio_inside_qrs / _outside_qrs / _composite (energy
    ratios, NOT F-wave preservation), TQ-specific metrics, valid_leads,
    extraction_status, notes and per_lead.

    F-wave preservation is NOT claimed from these ratios: on real ECG the
    original signal inside QRS contains both QRS and atrial activity. It must
    be validated on synthetic signals with a known atrial component.
    """
    sig = np.asarray(signal, dtype=float)
    n_leads_expected = len(lead_names)
    notes = []
    if sig.ndim != 2:
        raise ValueError("signal must be 2-D (leads, samples)")
    if sig.shape[0] != n_leads_expected and sig.shape[1] == n_leads_expected:
        sig = sig.T
        notes.append("auto-transposed (samples, leads) -> (leads, samples)")
    if sig.shape[0] != n_leads_expected:
        raise ValueError(f"lead/signal mismatch: {sig.shape[0]} rows vs {n_leads_expected} lead names")

    peaks = np.sort(np.asarray([] if qrs_peaks is None else qrs_peaks, dtype=int))
    residual = np.full_like(sig, np.nan)
    qrs_excl = np.zeros(sig.shape[1], dtype=bool)
    valid_mask = np.zeros(sig.shape[1], dtype=bool)
    per_lead = {}
    valid = []
    idx = _lead_index(lead_names)
    tq_segments = []

    for name in ("II", "V1", "III", "aVF", "V5"):
        if name not in idx:
            continue
        li = idx[name]
        x = sig[li]
        if method == "tq_segment":
            resid, tq_mask = _tq_residual(x, fs, peaks)
            qmask = _qrs_window_mask(len(x), fs, peaks)
            vmask = tq_mask
        else:
            tmpl = _beat_template(x, fs, peaks, method)
            if tmpl is None or tmpl[0] is None:
                resid = np.full_like(x, np.nan)
                qmask = np.zeros(len(x), dtype=bool)
                vmask = np.zeros(len(x), dtype=bool)
            else:
                template, L1, L2, _n = tmpl
                resid, qmask = _subtract_template(x, fs, peaks, template, L1, L2)
                vmask = np.isfinite(resid)
        ok = vmask.sum() > fs
        entry = {"n_beats_subtracted": 0, "qrs_suppression_quality": np.nan,
                 "residual_energy_ratio_inside_qrs": np.nan,
                 "residual_energy_ratio_outside_qrs": np.nan}
        if ok:
            residual[li] = resid
            qrs_excl |= qmask
            valid_mask |= vmask
            valid.append(name)
            if method == "tq_segment":
                segs = _mask_segments(tq_mask)
                tq_segments.extend((s / fs for s in segs))
                entry["n_beats_subtracted"] = int(len(segs))
            else:
                xa = _bandpass(x, fs, 3, 12)
                ra = _bandpass(np.nan_to_num(resid), fs, 3, 12)
                inside, outside = qmask, ~qmask
                pre_q = float(np.mean(x[inside] ** 2)) if inside.sum() else np.nan
                post_q = float(np.mean(np.nan_to_num(resid)[inside] ** 2)) if inside.sum() else np.nan
                entry["qrs_suppression_quality"] = (
                    float(np.clip(1.0 - post_q / (pre_q + 1e-12), 0.0, 1.0))
                    if np.isfinite(pre_q) and pre_q > 0 else np.nan)
                entry["residual_energy_ratio_inside_qrs"] = (
                    float(np.mean(ra[inside] ** 2) / (np.mean(xa[inside] ** 2) + 1e-12))
                    if inside.sum() else np.nan)
                entry["residual_energy_ratio_outside_qrs"] = (
                    float(np.mean(ra[outside] ** 2) / (np.mean(xa[outside] ** 2) + 1e-12))
                    if outside.sum() else np.nan)
                entry["n_beats_subtracted"] = int(len(peaks) if qmask.sum() else 0)
        per_lead[name] = entry

    def aggregated(key):
        vals = [v[key] for v in per_lead.values() if np.isfinite(v[key])]
        return float(np.mean(vals)) if vals else np.nan

    supp = aggregated("qrs_suppression_quality")
    ratio_in = aggregated("residual_energy_ratio_inside_qrs")
    ratio_out = aggregated("residual_energy_ratio_outside_qrs")
    ratio_composite = (float(np.sqrt(max(ratio_in, 1e-6) * max(ratio_out, 1e-6)))
                       if np.isfinite(ratio_in) and np.isfinite(ratio_out) else np.nan)

    if method == "tq_segment":
        valid_frac = float(valid_mask.mean()) if len(valid_mask) else 0.0
        valid_dur = float(valid_mask.sum() / fs)
        med_len = float(np.median(tq_segments)) if tq_segments else np.nan
        tq_status = ("ok" if valid_frac >= 0.3 and len(tq_segments) >= 3 else
                     "degraded" if valid_frac >= 0.1 else "failed")
        tq_metrics = {
            "valid_tq_duration": valid_dur,
            "valid_tq_fraction": valid_frac,
            "n_valid_tq_segments": int(len(tq_segments)),
            "median_tq_segment_length": med_len,
            "tq_extraction_status": tq_status,
        }
        supp = np.nan  # TQ excludes QRS; it does not cancel a template
        status = tq_status if valid else "failed"
    else:
        tq_metrics = {
            "valid_tq_duration": np.nan, "valid_tq_fraction": np.nan,
            "n_valid_tq_segments": 0, "median_tq_segment_length": np.nan,
            "tq_extraction_status": "not_applicable",
        }
        status = ("failed" if not valid or len(peaks) < 2 else
                  "degraded" if np.isfinite(supp) and supp < 0.3 else "ok")

    out = {
        "residual": residual,
        "qrs_exclusion_mask": qrs_excl,
        "valid_sample_mask": valid_mask,
        "qrs_suppression_quality": supp,
        "residual_energy_ratio_inside_qrs": ratio_in,
        "residual_energy_ratio_outside_qrs": ratio_out,
        "residual_energy_ratio_composite": ratio_composite,
        "valid_leads": valid,
        "extraction_status": status,
        "notes": notes,
        "per_lead": per_lead,
        "method": method,
        "n_qrs_peaks": int(len(peaks)),
    }
    out.update(tq_metrics)
    return out
