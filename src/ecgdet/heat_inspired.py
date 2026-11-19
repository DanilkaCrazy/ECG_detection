"""HEAT-inspired AV-conduction modelling for AFIB vs AFL.

Inspired by Sager et al. (2021), "Expert-enhanced machine learning for cardiac
arrhythmia classification" (PLOS ONE). This is a HEAT-INSPIRED approximation,
NOT an exact reproduction: the original uses a multi-level AV-block model with
Markov transitions fitted on 22 RR intervals. Here we search over a grid of
atrial cycle lengths and physiologically plausible conduction ratios, simulate
expected ventricular intervals, compare them to observed RR and evaluate
solution stability across time sub-windows (limited by 10-s ECG length).
"""

from __future__ import annotations

import numpy as np

__all__ = ["heat_features", "HEAT_RATIO_SETS"]

HEAT_RATIO_SETS = [
    (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4), (1, 2, 3, 4),
]
AC_GRID = np.arange(180, 361, 2)  # atrial cycle 180-360 ms -> 167-333 bpm


def _fit_heat(rr: np.ndarray, ac_grid=AC_GRID, ratio_sets=HEAT_RATIO_SETS):
    """Return best parameters and error for one RR sequence."""
    best = None
    for ratios in ratio_sets:
        mults = np.array(ratios, dtype=float)
        for ac in ac_grid:
            expected = mults[:, None] * ac  # (k,1)
            err = np.abs(rr[None, :] - expected)  # (k, n)
            chosen = np.argmin(err, axis=0)
            resid = err[chosen, np.arange(len(rr))]
            score = float(np.sqrt(np.mean(resid ** 2)))
            if best is None or score < best[0]:
                best = (score, float(ac), ratios, chosen, resid)
    return best


def heat_features(rr_ms: np.ndarray) -> dict:
    """Extract HEAT-inspired features from an RR interval sequence (ms)."""
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 300) & (rr < 2000)]
    out: dict[str, float] = {}
    if len(rr) < 3:
        for k in ("heat_best_rmse", "heat_best_ac", "heat_best_ratio", "heat_best_n_ratios",
                  "heat_explained_frac", "heat_ratio_entropy", "heat_ac_stability",
                  "heat_ratio_stability", "heat_best_rmse_rel"):
            out[k] = np.nan
        return out

    best = _fit_heat(rr)
    score, ac, ratios, chosen, resid = best
    out["heat_best_rmse"] = float(score)
    out["heat_best_rmse_rel"] = float(score / (np.mean(rr) + 1e-9))
    out["heat_best_ac"] = float(ac)
    out["heat_best_ratio"] = float(np.mean(np.array(ratios)[chosen]))
    out["heat_best_n_ratios"] = float(len(ratios))
    out["heat_explained_frac"] = float(np.mean(resid < 0.10 * rr))
    # entropy of chosen conduction ratios
    vals, cnt = np.unique(chosen, return_counts=True)
    p = cnt / cnt.sum()
    out["heat_ratio_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))

    # stability across two halves (if enough beats)
    if len(rr) >= 6:
        b1 = _fit_heat(rr[: len(rr) // 2])
        b2 = _fit_heat(rr[len(rr) // 2 :])
        out["heat_ac_stability"] = float(abs(b1[1] - b2[1]))
        out["heat_ratio_stability"] = float(abs(np.mean(np.array(b1[2])[b1[3]]) -
                                                 np.mean(np.array(b2[2])[b2[3]])))
    else:
        out["heat_ac_stability"] = np.nan
        out["heat_ratio_stability"] = np.nan
    return out
