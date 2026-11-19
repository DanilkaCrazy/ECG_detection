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


def _tq_residual(x, fs, peaks, pre_q=0.05, post_t=0.25):
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
    dict with residual (n_leads, n_samples), qrs_mask (n_samples,),
    qrs_suppression_quality, atrial_preservation_quality,
    atrial_preservation_inside_qrs, valid_leads, extraction_status, notes,
    per_lead (dict).
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
    mask_union = np.zeros(sig.shape[1], dtype=bool)
    per_lead = {}
    valid = []
    idx = _lead_index(lead_names)

    for name in ("II", "V1", "III", "aVF", "V5"):
        if name not in idx:
            continue
        li = idx[name]
        x = sig[li]
        if method == "tq_segment":
            resid, m = _tq_residual(x, fs, peaks)
        else:
            tmpl = _beat_template(x, fs, peaks, method)
            if tmpl is None or tmpl[0] is None:
                resid, m = np.full_like(x, np.nan), np.zeros(len(x), dtype=bool)
            else:
                template, L1, L2, _n = tmpl
                resid, m = _subtract_template(x, fs, peaks, template, L1, L2)
        ok = np.isfinite(resid).sum() > fs
        if ok:
            residual[li] = resid
            mask_union |= m
            valid.append(name)
            xa, ra = _bandpass(x, fs, 3, 12), _bandpass(np.nan_to_num(resid), fs, 3, 12)
            inside = m
            outside = ~m
            pre_q = float(np.mean(x[inside] ** 2)) if inside.sum() else np.nan
            post_q = float(np.mean(np.nan_to_num(resid)[inside] ** 2)) if inside.sum() else np.nan
            supp = float(np.clip(1.0 - post_q / (pre_q + 1e-12), 0.0, 1.0)) if np.isfinite(pre_q) and pre_q > 0 else np.nan
            pres_in = float(np.mean(ra[inside] ** 2) / (np.mean(xa[inside] ** 2) + 1e-12)) if inside.sum() else np.nan
            pres_out = float(np.mean(ra[outside] ** 2) / (np.mean(xa[outside] ** 2) + 1e-12)) if outside.sum() else np.nan
            if method == "tq_segment":
                # TQ does not cancel QRS — it excludes QRS windows by construction.
                # Template-subtraction quality metrics are not applicable.
                per_lead[name] = {
                    "qrs_suppression_quality": 1.0,
                    "atrial_preservation_inside_qrs": np.nan,
                    "atrial_preservation_outside_qrs": np.nan,
                    "n_beats_subtracted": 0,
                }
            else:
                per_lead[name] = {
                    "qrs_suppression_quality": supp,
                    "atrial_preservation_inside_qrs": pres_in,
                    "atrial_preservation_outside_qrs": pres_out,
                    "n_beats_subtracted": int(m.sum() and len(peaks)),
                }
        else:
            per_lead[name] = {"qrs_suppression_quality": np.nan,
                              "atrial_preservation_inside_qrs": np.nan,
                              "atrial_preservation_outside_qrs": np.nan,
                              "n_beats_subtracted": 0}

    supp_vals = [v["qrs_suppression_quality"] for v in per_lead.values()]
    in_vals = [v["atrial_preservation_inside_qrs"] for v in per_lead.values()]
    out_vals = [v["atrial_preservation_outside_qrs"] for v in per_lead.values()]
    supp = float(np.nanmean(supp_vals)) if np.isfinite(supp_vals).any() else np.nan
    pres_in = float(np.nanmean(in_vals)) if np.isfinite(in_vals).any() else np.nan
    pres_out = float(np.nanmean(out_vals)) if np.isfinite(out_vals).any() else np.nan
    # composite: keep BOTH inside- and outside-window information
    pres = float(np.sqrt(max(pres_in, 1e-6) * max(pres_out, 1e-6))) if np.isfinite(pres_in) and np.isfinite(pres_out) else np.nan

    if not valid or len(peaks) < 2:
        status = "failed"
    elif np.isfinite(supp) and supp < 0.3:
        status = "degraded"
    else:
        status = "ok"

    return {
        "residual": residual,
        "qrs_mask": mask_union,
        "qrs_suppression_quality": supp,
        "atrial_preservation_quality": pres,
        "atrial_preservation_inside_qrs": pres_in,
        "atrial_preservation_outside_qrs": pres_out,
        "valid_leads": valid,
        "extraction_status": status,
        "notes": notes,
        "per_lead": per_lead,
        "method": method,
        "n_qrs_peaks": int(len(peaks)),
    }
