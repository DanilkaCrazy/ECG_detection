"""Stage 3.5 — advanced atrial signal analysis (P1).

- multi-lead QRS detection robust to negative polarity;
- three QRST cancellation strategies with measured QRS suppression quality;
- spectral features (prominence, bandwidth, entropy, harmonics, stability);
- temporal organization and multilead coherence;
- explicit lead-name based channel mapping (no blind fixed indices).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

__all__ = ["lead_index_map", "multi_lead_qrs", "qrs_cancel_compare",
           "spectral_atrial", "temporal_atrial", "multilead_coherence"]

_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "MLII": "II"}


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
        b, a = sp_signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return sp_signal.filtfilt(b, a, np.asarray(x, dtype=float))
    except Exception:
        return np.asarray(x, dtype=float)


def multi_lead_qrs(sig2d: np.ndarray, fs: float, lead_names: list[str]) -> tuple[np.ndarray, dict]:
    """Consensus QRS detection using II, V1, V5 (or available), polarity-robust."""
    idx = lead_index_map(lead_names)
    chosen = [idx[l] for l in ("II", "V1", "V5") if l in idx]
    if not chosen:
        chosen = list(range(min(3, sig2d.shape[0])))
    votes = np.zeros(sig2d.shape[1], dtype=float)
    per_lead = {}
    for ci in chosen:
        x = _bandpass(sig2d[ci], fs, 5, 15)
        # polarity robustness: rectify with squared signal
        energy = x ** 2
        thr = np.mean(energy) + 1.5 * np.std(energy)
        peaks, _ = sp_signal.find_peaks(energy, height=thr, distance=int(0.25 * fs),
                                        prominence=max(thr * 0.2, 1e-12))
        per_lead[ci] = peaks
        for p in peaks:
            votes[max(0, p - int(0.03 * fs)):min(len(votes), p + int(0.03 * fs))] += 1
    # consensus peaks where >= 2 leads voted (or >=1 if only one lead)
    need = 2 if len(chosen) > 1 else 1
    cand, _ = sp_signal.find_peaks(votes, height=need - 0.5, distance=int(0.25 * fs))
    # refine to the strongest local vote position
    refined = []
    for c in cand:
        lo, hi = max(0, c - int(0.04 * fs)), min(len(votes), c + int(0.04 * fs))
        refined.append(int(lo + np.argmax(votes[lo:hi])))
    peaks = np.unique(np.asarray(refined, dtype=int))
    return peaks, {"consensus_leads": chosen, "votes": len(cand),
                   "per_lead_counts": {str(k): int(len(v)) for k, v in per_lead.items()}}


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
        corr = np.corrcoef(B)
        ref = int(np.argmax(np.sum(corr, axis=1)))
        good = corr[ref] > 0.6
        template = np.median(B[good], axis=0) if good.sum() >= 2 else np.median(B, axis=0)
    else:  # beat_aligned
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
    """Compare cancellations; report measured QRS suppression + preservation."""
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
            # preservation: atrial-band energy outside QRS windows retained
            xb, rb = _bandpass(x, fs, 3, 12), _bandpass(resid, fs, 3, 12)
            outside = ~mask
            if outside.sum() > 10:
                preservation = float(np.mean(rb[outside] ** 2) / (np.mean(xb[outside] ** 2) + 1e-12))
            else:
                preservation = np.nan
            per_lead.append({"lead": l, "qrs_residual_energy": post / (pre + 1e-12),
                             "preservation": preservation,
                             "resid_std": float(np.std(resid))})
        if per_lead:
            # aggregate dict plus per-lead detail
            detail = {d["lead"]: d for d in per_lead}
            result[method] = {
                "qrs_residual_energy": float(np.nanmean([d["qrs_residual_energy"] for d in per_lead])),
                "preservation": float(np.nanmean([d["preservation"] for d in per_lead])),
                "resid_std": float(np.nanmean([d["resid_std"] for d in per_lead])),
                "per_lead": detail,
            }
        else:
            result[method] = {"qrs_residual_energy": np.nan, "preservation": np.nan,
                              "resid_std": np.nan, "per_lead": {}}
    return result


def spectral_atrial(x: np.ndarray, fs: float, lo: float = 3.0, hi: float = 10.0) -> dict:
    x = _bandpass(x, fs, lo, hi)
    if len(x) < 256:
        return {k: np.nan for k in ("dominant_freq", "peak_prominence", "peak_to_background",
                                    "spectral_entropy", "spectral_bandwidth", "peak_width",
                                    "harmonic_ratio", "dominant_freq_stability")}
    nperseg = min(1024, len(x))
    f, p = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    m = (f >= lo) & (f <= hi)
    pf, pp = f[m], p[m]
    j = int(np.argmax(pp))
    # prominence via local background
    bg = float(np.median(pp)) + 1e-12
    pk = float(pp[j])
    # width at half prominence above background
    half = bg + (pk - bg) / 2
    above = np.where(pp >= half)[0]
    width = float(f[above[-1]] - f[above[0]]) / (hi - lo) if len(above) > 1 else 0.0
    tot = pp.sum() + 1e-12
    pr = pp / tot
    # harmonic structure: energy at 2x dominant vs dominant
    harm = (f >= min(2 * pf[j], hi)) & (f <= hi)
    out = {
        "dominant_freq": float(pf[j]),
        "peak_prominence": float(pk - bg),
        "peak_to_background": float(pk / bg),
        "spectral_entropy": float(-np.sum(pr * np.log(pr + 1e-12))),
        "spectral_bandwidth": float(np.sqrt(np.sum(pr * (pf - np.sum(pr * pf)) ** 2))),
        "spectral_concentration": float(pk / tot),
        "peak_width": width,
        "harmonic_ratio": float(np.max(p[harm]) / (pk + 1e-12)) if harm.any() else 0.0,
    }
    # stability across 3 windows
    dom = []
    for part in np.array_split(x, 3):
        if len(part) < 128:
            continue
        fp, pp2 = sp_signal.welch(part, fs=fs, nperseg=min(512, len(part)))
        mm = (fp >= lo) & (fp <= hi)
        if mm.any():
            dom.append(float(fp[mm][int(np.argmax(pp2[mm]))]))
    out["dominant_freq_stability"] = float(np.std(dom)) if len(dom) >= 2 else np.nan
    return out


def temporal_atrial(x: np.ndarray, fs: float, peaks: np.ndarray) -> dict:
    out = {}
    if len(peaks) >= 3:
        cyc = np.diff(peaks) / fs * 1000
        cyc = cyc[(cyc > 150) & (cyc < 400)]
        if len(cyc) >= 2:
            out["atrial_cycle_mean"] = float(np.mean(cyc))
            out["atrial_cycle_cv"] = float(np.std(cyc) / (np.mean(cyc) + 1e-9))
        else:
            out.update({"atrial_cycle_mean": np.nan, "atrial_cycle_cv": np.nan})
    else:
        out.update({"atrial_cycle_mean": np.nan, "atrial_cycle_cv": np.nan})
    x = _bandpass(x, fs, 3, 12)
    if len(x) > 512:
        r = x - x.mean()
        ac = sp_signal.correlate(r, r, mode="full")[len(r)-1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(0.171 * fs), min(int(0.4 * fs), len(ac)-1)
        out["periodicity"] = float(np.max(ac[lo:hi])) if hi > lo else np.nan
    else:
        out["periodicity"] = np.nan
    # cycle-to-cycle morphology consistency
    L = int(0.2 * fs)
    cycles = []
    for p in peaks:
        a, b = p - L//4, p + L - L//4
        if a >= 0 and b <= len(x):
            c = x[a:b]
            cycles.append(c / (np.std(c) + 1e-9))
    if len(cycles) >= 2:
        C = np.asarray(cycles); corr = np.corrcoef(C); iu = np.triu_indices(len(C), 1)
        out["cycle_morph_consistency"] = float(np.nanmean(corr[iu]))
    else:
        out["cycle_morph_consistency"] = np.nan
    return out


def multilead_coherence(sig2d: np.ndarray, fs: float, lead_names: list[str]) -> dict:
    idx = lead_index_map(lead_names)
    leads = [(l, idx[l]) for l in ("II", "III", "aVF", "V1") if l in idx]
    if len(leads) < 2:
        return {"cross_lead_coherence_mean": np.nan, "cross_lead_coherence_std": np.nan}
    xs = np.asarray([_bandpass(sig2d[i], fs, 3, 12) for _, i in leads])
    C = np.corrcoef(xs)
    iu = np.triu_indices(len(leads), 1)
    return {"cross_lead_coherence_mean": float(np.nanmean(C[iu])),
            "cross_lead_coherence_std": float(np.nanstd(C[iu]))}
