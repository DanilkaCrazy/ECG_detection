"""Stage 3.6.20 — physiological RR / AV-conduction model features.

Inspired by the AV-block modelling idea in Sager et al. (PLOS ONE 2021) and the
prior art EP2572634A1, but implemented as an independently defined feature set:
for each candidate atrial cycle we test whether observed RR intervals are
integer multiples (2:1, 3:1, ... AV block) of that cycle.
"""
from __future__ import annotations

import numpy as np

__all__ = ["av_conduction_features", "FEATURE_NAMES"]

FEATURE_NAMES = (
    "rr_cv", "rr_rmssd", "rr_sdnn", "rr_sd1", "rr_sd2", "rr_sample_entropy",
    "rr_autocorr_peak_lag", "atrial_cycle_s", "av_ratio_consistency",
    "av_model_error", "av_n_levels", "multi_level_block_prob",
)


def _sample_entropy(x, m=2, r_factor=0.2):
    if len(x) < m + 2:
        return 0.0
    r = r_factor * float(np.std(x))
    if r <= 1e-12:
        return 0.0

    def count(t):
        total = 0
        for i in range(len(x) - t + 1):
            a = x[i:i + t]
            for j in range(i + 1, len(x) - t + 1):
                if np.max(np.abs(a - x[j:j + t])) <= r:
                    total += 1
        return total

    a, b = count(m), count(m + 1)
    return float(-np.log(b / a)) if a and b else 0.0


def av_conduction_features(rr_intervals: np.ndarray) -> dict[str, float]:
    """Fit an explicit AV-conduction model to an RR sequence (seconds)."""
    rr = np.asarray(rr_intervals, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 3:
        return {k: 0.0 for k in FEATURE_NAMES}

    mean, sd = float(rr.mean()), float(np.std(rr))
    diff = np.diff(rr)
    cv = sd / mean if mean > 1e-12 else 0.0
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    sd1 = float(np.std(diff) / np.sqrt(2))
    ent = _sample_entropy(rr)

    if rr.size >= 6 and sd > 1e-12:
        z = rr - mean
        ac = np.correlate(z, z, mode="full")[len(z) - 1:]
        ac = ac / (np.dot(z, z) + 1e-12)
        lags = np.arange(1, min(20, len(z) // 2))
        peak_lag = int(lags[np.argmax(ac[lags])]) if len(lags) else 0
    else:
        peak_lag = 0

    # Exclude the trivial "atrial cycle == mean RR" solution (1:1), which is
    # not an AV block and makes the model error artificially tiny.
    upper = min(0.60, float(np.median(rr)) / 1.5)
    candidates = np.linspace(0.18, max(0.181, upper), 85)
    scores = []
    for cyc in candidates:
        ratio = np.clip(np.rint(rr / cyc), 1, 6)
        residual = np.abs(rr - ratio * cyc) / cyc
        scores.append((float(np.mean(residual)), float(np.mean(ratio))))
    scores = np.asarray(scores)
    best_score = scores[:, 0].min()
    # Tie-break toward the smallest atrial cycle (highest AV ratio), which is
    # the physiologically meaningful solution for flutter with 2:1/3:1 block.
    near = np.where(scores[:, 0] <= best_score + 1e-3)[0]
    chosen = int(near[np.argmax(scores[near, 1])])
    cycle = float(candidates[chosen])
    ratio = np.clip(np.rint(rr / cycle), 1, 6)
    residual = np.abs(rr - ratio * cycle) / cycle
    consistency = float(np.mean(residual < 0.05))
    counts = np.bincount(ratio.astype(int), minlength=7) / len(rr)
    n_levels = int(np.sum(counts > 0.05))
    multi = float(np.mean(ratio >= 2))

    return {
        "rr_cv": float(cv), "rr_rmssd": rmssd, "rr_sdnn": sd,
        "rr_sd1": sd1, "rr_sd2": sd,
        "rr_sample_entropy": ent,
        "rr_autocorr_peak_lag": float(peak_lag),
        "atrial_cycle_s": cycle,
        "av_ratio_consistency": consistency,
        "av_model_error": float(best_score),
        "av_n_levels": float(n_levels),
        "multi_level_block_prob": multi,
    }
