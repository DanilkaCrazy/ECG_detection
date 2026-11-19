"""Stage 3.6 — advanced atrial signal analysis.

Corrected relative to Stage 3.5:
- spectral features are computed on the QRST-residual supplied by the caller;
- peak width is CONTIGUOUS around the dominant peak;
- harmonic energy is measured near 2f and 3f with tolerance;
- Hoppe-style peak area ratios and spectral flatness are included;
- temporal analysis never uses QRS R-peaks as atrial cycle peaks and searches
  the full 3-10 Hz autocorrelation range;
- multilead analysis uses magnitude-squared coherence and phase locking value
  (Pearson correlation is retained only as a legacy comparison field);
- qrs_cancel_compare reports preservation INSIDE the QRS windows.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

__all__ = ["lead_index_map", "multi_lead_qrs", "qrs_cancel_compare",
           "spectral_atrial", "temporal_atrial", "multilead_coherence"]

_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "MLII": "II"}
_NAN_KEYS = ("dominant_freq", "peak_prominence", "peak_to_background",
             "spectral_entropy", "spectral_bandwidth", "spectral_concentration",
             "peak_width", "harmonic_ratio", "harmonic_2f_ratio", "harmonic_3f_ratio",
             "spectral_flatness", "peak_sharpness_db", "spectral_peak_isolation",
             "dominant_freq_stability", "dominant_freq_cv",
             "peak_area_ratio_0_625hz", "peak_area_ratio_1_25hz", "peak_area_ratio_2_5hz",
             "peak_area_ratio_3_75hz", "peak_area_ratio_5hz")


def lead_index_map(lead_names: list[str]) -> dict[str, int]:
    out = {}
    for i, n in enumerate(lead_names):
        key = str(n).strip().upper()
        key = _ALIASES.get(key, key)
        if key not in out:
            out[key] = i
    return out


def _bandpass(x, fs, lo, hi):
    try:
        nyq = fs / 2.0
        if hi >= nyq or lo <= 0 or len(x) < 8:
            return np.asarray(x, dtype=float).copy()
        b, a = sp_signal.butter(3, [lo / nyq, hi / nyq], btype="band")
        return sp_signal.filtfilt(b, a, np.asarray(x, dtype=float))
    except Exception:
        return np.asarray(x, dtype=float)


def multi_lead_qrs(sig2d: np.ndarray, fs: float, lead_names: list[str]) -> tuple[np.ndarray, dict]:
    """Consensus QRS detection using II, V1, V5 (or available), polarity-robust.

    Expects (leads, samples). A transposed input is auto-detected and transposed;
    Stage 3.5 silently mis-used (samples, leads) and detected zero peaks.
    """
    sig2d = np.asarray(sig2d)
    if sig2d.ndim != 2:
        raise ValueError("signal must be (leads, samples)")
    if sig2d.shape[0] != len(lead_names) and sig2d.shape[1] == len(lead_names):
        sig2d = sig2d.T
    idx = lead_index_map(lead_names)
    chosen = [idx[l] for l in ("II", "V1", "V5") if l in idx]
    if not chosen:
        chosen = list(range(min(3, sig2d.shape[0])))
    votes = np.zeros(sig2d.shape[1], dtype=float)
    per_lead = {}
    for ci in chosen:
        x = _bandpass(sig2d[ci], fs, 5, 15)
        energy = x ** 2
        # Robust threshold: mean+std is dominated by a single artifact or by
        # QRS complexes themselves on near-zero-baseline sources (PTB-XL),
        # leaving a threshold no other complex can cross.
        med_e = float(np.median(energy))
        mad_e = float(np.median(np.abs(energy - med_e))) * 1.4826
        thr = max(med_e + 3.0 * mad_e, 1e-12)
        peaks, _ = sp_signal.find_peaks(energy, height=thr, distance=int(0.25 * fs),
                                        prominence=max(thr * 0.2, 1e-12))
        per_lead[ci] = peaks
        for p in peaks:
            votes[max(0, p - int(0.03 * fs)):min(len(votes), p + int(0.03 * fs))] += 1
    need = 2 if len(chosen) > 1 else 1
    cand, _ = sp_signal.find_peaks(votes, height=need - 0.5, distance=int(0.25 * fs))
    refined = []
    for c in cand:
        lo, hi = max(0, c - int(0.04 * fs)), min(len(votes), c + int(0.04 * fs))
        refined.append(int(lo + np.argmax(votes[lo:hi])))
    peaks = np.unique(np.asarray(refined, dtype=int))
    return peaks, {"consensus_leads": chosen, "votes": len(cand),
                   "per_lead_counts": {str(k): int(len(v)) for k, v in per_lead.items()},
                   "auto_transposed": bool(sig2d.shape[0] == len(lead_names))}


def _cancel(x, fs, peaks, method, pre=0.08, post=0.12):
    x = np.asarray(x, dtype=float)
    L1, L2 = int(pre * fs), int(post * fs)
    L = L1 + L2
    beats = [x[p - L1:p + L2] for p in peaks if p - L1 >= 0 and p + L2 <= len(x)]
    if len(beats) < 2:
        return x.copy(), np.zeros(len(x), dtype=bool)
    B = np.asarray(beats)
    if method == "median":
        template = np.median(B, axis=0)
    elif method == "adaptive":
        corr = np.nan_to_num(np.corrcoef(B))
        ref = int(np.argmax(np.sum(corr, axis=1)))
        good = corr[ref] > 0.6
        template = np.median(B[good], axis=0) if good.sum() >= 2 else np.median(B, axis=0)
    else:
        template = np.mean(B, axis=0)
    w = np.hanning(L)
    out = x.copy(); mask = np.zeros(len(x), dtype=bool)
    for p in peaks:
        lo, hi = p - L1, p + L2
        if lo >= 0 and hi <= len(x):
            out[lo:hi] -= template * w
            mask[lo:hi] = True
    return out, mask


def qrs_cancel_compare(sig2d: np.ndarray, fs: float, lead_names: list[str],
                       peaks: np.ndarray) -> dict:
    """Compare cancellations; report measured QRS suppression + preservation.

    Adds ``atrial_preservation_inside_qrs`` measured INSIDE QRS windows.
    """
    sig2d = np.asarray(sig2d)
    if sig2d.shape[0] != len(lead_names) and sig2d.shape[1] == len(lead_names):
        sig2d = sig2d.T
    idx = lead_index_map(lead_names)
    leads = [l for l in ("II", "V1") if l in idx]
    result = {}
    for method in ("median", "adaptive", "beat_aligned"):
        per_lead = []
        for l in leads:
            x = sig2d[idx[l]]
            resid, mask = _cancel(x, fs, peaks, method)
            if mask.sum() == 0:
                continue
            pre = float(np.mean(x[mask] ** 2))
            post = float(np.mean(resid[mask] ** 2))
            xb, rb = _bandpass(x, fs, 3, 12), _bandpass(resid, fs, 3, 12)
            inside, outside = mask, ~mask
            pres_in = float(np.mean(rb[inside] ** 2) / (np.mean(xb[inside] ** 2) + 1e-12))
            pres_out = float(np.mean(rb[outside] ** 2) / (np.mean(xb[outside] ** 2) + 1e-12)) if outside.sum() > 10 else np.nan
            per_lead.append({"lead": l, "qrs_residual_energy": post / (pre + 1e-12),
                             "preservation": pres_out,
                             "atrial_preservation_inside_qrs": pres_in,
                             "resid_std": float(np.std(resid))})
        if per_lead:
            detail = {d["lead"]: d for d in per_lead}
            result[method] = {
                "qrs_residual_energy": float(np.nanmean([d["qrs_residual_energy"] for d in per_lead])),
                "preservation": float(np.nanmean([d["preservation"] for d in per_lead])),
                "atrial_preservation_inside_qrs": float(np.nanmean([d["atrial_preservation_inside_qrs"] for d in per_lead])),
                "resid_std": float(np.nanmean([d["resid_std"] for d in per_lead])),
                "per_lead": detail,
            }
        else:
            result[method] = {"qrs_residual_energy": np.nan, "preservation": np.nan,
                              "atrial_preservation_inside_qrs": np.nan,
                              "resid_std": np.nan, "per_lead": {}}
    return result


def _contiguous_width(freqs, psd, j, frac=0.5):
    """Width of the contiguous region around peak j above background + fraction."""
    bg = float(np.median(psd)) + 1e-12
    pk = float(psd[j])
    thr = bg + (pk - bg) * frac
    lo = j
    while lo > 0 and psd[lo - 1] >= thr:
        lo -= 1
    hi = j
    while hi < len(psd) - 1 and psd[hi + 1] >= thr:
        hi += 1
    return float(freqs[hi] - freqs[lo]), lo, hi


def _band_energy(f, p, center, tol):
    m = (f >= center - tol) & (f <= center + tol)
    return float(np.sum(p[m])) if m.any() else 0.0


def _welch_psd_with_gaps(x, fs, max_nperseg=4096):
    """Welch PSD; for NaN-gapped TQ residuals average per contiguous segment.

    Concatenating TQ segments breaks phase and produces spurious peaks, so each
    valid segment gets its own periodogram and the PSDs are averaged.
    """
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.all():
        return sp_signal.welch(x, fs=fs, nperseg=min(max_nperseg, len(x)),
                               nfft=max(8192, len(x)))
    segs, i = [], 0
    while i < len(x):
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(x) and valid[j + 1]:
            j += 1
        if j - i + 1 >= 64:
            segs.append(x[i:j + 1])
        i = j + 1
    if not segs:
        return None, None
    # fixed nperseg -> identical frequency axes -> PSDs can be averaged
    nperseg = min(max_nperseg, min(len(s) for s in segs))
    psds = []
    for s in segs:
        f, p = sp_signal.welch(s, fs=fs, nperseg=nperseg, nfft=max(8192, nperseg))
        psds.append(p)
    return f, np.mean(psds, axis=0)


def spectral_atrial(x: np.ndarray, fs: float, lo: float = 3.0, hi: float = 10.0) -> dict:
    """Spectral features of an atrial residual (caller must supply residual)."""
    x = np.asarray(x, dtype=float)
    f, p = _welch_psd_with_gaps(x, fs)
    if f is None or len(p) < 4:
        return {k: np.nan for k in _NAN_KEYS}
    xf = np.nan_to_num(x)
    xf = _bandpass(xf, fs, lo, hi)
    if len(xf) < 256:
        return {k: np.nan for k in _NAN_KEYS}
    m = (f >= lo) & (f <= hi)
    pf, pp = f[m], p[m]
    if len(pp) < 4:
        return {k: np.nan for k in _NAN_KEYS}
    j = int(np.argmax(pp))
    bg = float(np.median(pp)) + 1e-12
    pk = float(pp[j])
    width, wlo, whi = _contiguous_width(pf, pp, j)
    tot = pp.sum() + 1e-12
    pr = pp / tot
    df = float(pf[1] - pf[0]) if len(pf) > 1 else 0.25
    # Hoppe-style peak area ratios: peak band / surrounding area.
    ratios = {}
    for halfwidth in (0.625, 1.25, 2.5, 3.75, 5.0):
        center = float(pf[j])
        band = _band_energy(pf, pp, center, halfwidth)
        left = _band_energy(pf, pp, center - 2 * halfwidth, halfwidth)
        right = _band_energy(pf, pp, center + 2 * halfwidth, halfwidth)
        ratios[halfwidth] = float(band / (band + left + right + 1e-12))
    # harmonics around 2f and 3f with tolerance. Both numerator and denominator
    # are band energies with the SAME tolerance -> interpretable power ratio.
    fd = float(pf[j])
    ht = max(2 * df, 0.35)
    fund_energy = _band_energy(pf, pp, fd, ht)
    h2 = _band_energy(pf, pp, 2 * fd, ht) / (fund_energy + 1e-12)
    h3 = _band_energy(pf, pp, 3 * fd, ht) / (fund_energy + 1e-12)
    # spectral flatness (Wiener entropy) in atrial band
    flat = float(np.exp(np.mean(np.log(pp + 1e-12))) / (np.mean(pp) + 1e-12))
    # sharpness in dB: peak / mean of the contiguous wing region
    wing = np.concatenate([pp[max(0, wlo - 6):wlo], pp[whi + 1:whi + 7]])
    sharp = float(10 * np.log10((pk + 1e-12) / (np.mean(wing) + 1e-12))) if wing.size else np.nan
    # isolation: peak power / total power outside its contiguous region
    outside = np.concatenate([pp[:wlo], pp[whi + 1:]])
    isolation = float(pk / (np.sum(outside) + 1e-12)) if outside.size else np.nan
    out = {
        "dominant_freq": fd,
        "peak_prominence": float(pk - bg),
        "peak_to_background": float(pk / bg),
        "spectral_entropy": float(-np.sum(pr * np.log(pr + 1e-12))),
        "spectral_bandwidth": float(np.sqrt(np.sum(pr * (pf - np.sum(pr * pf)) ** 2))),
        "spectral_concentration": float(pk / tot),
        "peak_width": width / max(hi - lo, 1e-9),
        "harmonic_ratio": float(h2),  # legacy alias
        "harmonic_2f_ratio": float(h2),
        "harmonic_3f_ratio": float(h3),
        "spectral_flatness": flat,
        "peak_sharpness_db": sharp,
        "spectral_peak_isolation": isolation,
    }
    for hw, val in ratios.items():
        key = {0.625: "0_625hz", 1.25: "1_25hz", 2.5: "2_5hz", 3.75: "3_75hz", 5.0: "5hz"}[hw]
        out[f"peak_area_ratio_{key}"] = val
    dom = []
    for part in np.array_split(xf, 3):
        if len(part) < 128:
            continue
        fp, pp2 = sp_signal.welch(part, fs=fs, nperseg=min(512, len(part)))
        mm = (fp >= lo) & (fp <= hi)
        if mm.any():
            dom.append(float(fp[mm][int(np.argmax(pp2[mm]))]))
    out["dominant_freq_stability"] = float(np.std(dom)) if len(dom) >= 2 else np.nan
    out["dominant_freq_cv"] = float(np.std(dom) / (np.mean(dom) + 1e-9)) if len(dom) >= 2 else np.nan
    return out


def temporal_atrial(x: np.ndarray, fs: float, peaks: np.ndarray,
                    qrs_peaks: np.ndarray | None = None) -> dict:
    """Temporal organisation of atrial activity.

    ``peaks`` MUST be atrial-cycle peaks (from the residual), never QRS R-peaks.
    ``qrs_peaks`` is accepted only to report RR statistics separately.
    """
    out = {}
    ap = np.sort(np.asarray([] if peaks is None else peaks, dtype=int))
    if len(ap) >= 3:
        cyc = np.diff(ap) / fs * 1000
        cyc = cyc[(cyc > 60) & (cyc < 500)]  # 2-16 Hz atrial range
        out["atrial_cycle_mean"] = float(np.mean(cyc)) if len(cyc) else np.nan
        out["atrial_cycle_cv"] = float(np.std(cyc) / (np.mean(cyc) + 1e-9)) if len(cyc) else np.nan
    else:
        out.update({"atrial_cycle_mean": np.nan, "atrial_cycle_cv": np.nan})

    if qrs_peaks is not None and len(qrs_peaks) >= 2:
        rr = np.diff(np.sort(np.asarray(qrs_peaks))) / fs * 1000
        out["rr_mean"] = float(np.mean(rr))
        out["rr_cv"] = float(np.std(rr) / (np.mean(rr) + 1e-9))
        out["atrial_to_ventricular_rate"] = float(1000 / (out["atrial_cycle_mean"] + 1e-9) * (out["rr_mean"] / 1000)) if np.isfinite(out["atrial_cycle_mean"]) else np.nan
    else:
        out.update({"rr_mean": np.nan, "rr_cv": np.nan, "atrial_to_ventricular_rate": np.nan})

    x = _bandpass(x, fs, 3, 12)
    if len(x) > 512:
        r = x - x.mean()
        ac = sp_signal.correlate(r, r, mode="full")[len(r) - 1:]
        ac = ac / (ac[0] + 1e-12)
        # full declared 3-10 Hz band -> lags from fs/10 to fs/3
        lag_lo, lag_hi = max(1, int(fs / 10.0)), min(int(fs / 3.0), len(ac) - 1)
        out["periodicity"] = float(np.max(ac[lag_lo:lag_hi])) if lag_hi > lag_lo else np.nan
    else:
        out["periodicity"] = np.nan

    L = int(0.2 * fs)
    cycles = []
    for p in ap:
        a, b = p - L // 4, p + L - L // 4
        if a >= 0 and b <= len(x):
            c = x[a:b]
            cycles.append(c / (np.std(c) + 1e-9))
    if len(cycles) >= 2:
        C = np.asarray(cycles)
        corr = np.nan_to_num(np.corrcoef(C))
        iu = np.triu_indices(len(C), 1)
        out["cycle_morph_consistency"] = float(np.mean(corr[iu])) if len(corr) else np.nan
    else:
        out["cycle_morph_consistency"] = np.nan
    return out


def multilead_coherence(sig2d: np.ndarray, fs: float, lead_names: list[str]) -> dict:
    """Magnitude-squared coherence + phase locking value in the atrial band.

    Pearson correlation is kept under the old keys for backward compatibility
    only; it is NOT a coherence measure and must not be used as one.
    """
    sig2d = np.asarray(sig2d)
    if sig2d.shape[0] != len(lead_names) and sig2d.shape[1] == len(lead_names):
        sig2d = sig2d.T
    idx = lead_index_map(lead_names)
    leads = [(l, idx[l]) for l in ("II", "III", "aVF", "V1") if l in idx]
    if len(leads) < 2:
        return {"cross_spectral_coherence_mean": np.nan, "phase_locking_value_mean": np.nan,
                "cross_lead_coherence_mean": np.nan, "cross_lead_coherence_std": np.nan}
    xs = np.asarray([_bandpass(sig2d[i], fs, 3, 12) for _, i in leads])
    coh_vals, plv_vals = [], []
    for a in range(len(leads)):
        for b in range(a + 1, len(leads)):
            fa, Pa = sp_signal.welch(xs[a], fs=fs, nperseg=min(1024, len(xs[a])))
            fb, Pb = sp_signal.welch(xs[b], fs=fs, nperseg=min(1024, len(xs[b])))
            f, C = sp_signal.coherence(xs[a], xs[b], fs=fs, nperseg=min(1024, len(xs[a])))
            m = (f >= 3) & (f <= 10)
            if m.any():
                coh_vals.append(float(np.mean(np.abs(C[m]) ** 2)))
            # phase locking value from Hilbert transform of bandpassed signals
            za = sp_signal.hilbert(xs[a]); zb = sp_signal.hilbert(xs[b])
            phase_diff = np.angle(za) - np.angle(zb)
            plv_vals.append(float(np.abs(np.mean(np.exp(1j * phase_diff)))))
    C_legacy = np.corrcoef(xs)
    iu = np.triu_indices(len(leads), 1)
    return {
        "cross_spectral_coherence_mean": float(np.nanmean(coh_vals)) if coh_vals else np.nan,
        "cross_spectral_coherence_std": float(np.nanstd(coh_vals)) if coh_vals else np.nan,
        "phase_locking_value_mean": float(np.nanmean(plv_vals)) if plv_vals else np.nan,
        "cross_lead_coherence_mean": float(np.nanmean(C_legacy[iu])),
        "cross_lead_coherence_std": float(np.nanstd(C_legacy[iu])),
    }
