"""Stage 3.6.10 — Group A: ventricular rhythm / RR regularity features.

RR features alone cannot separate AFIB from AFL with variable AV conduction,
so they are one group among five, never a standalone classifier.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rhythm_features", "FEATURE_NAMES"]

FEATURE_NAMES = (
    "rr_cv", "rr_rmssd", "rr_entropy", "rr_sample_entropy",
    "rr_poincare_sd1", "rr_poincare_sd2", "rr_periodicity",
    "rr_irregularity_index",
)


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_factor * float(np.std(x))
    if r <= 1e-12:
        return 0.0

    def count(templates: int) -> int:
        total = 0
        for i in range(n - templates + 1):
            a = x[i:i + templates]
            for j in range(i + 1, n - templates + 1):
                if np.max(np.abs(a - x[j:j + templates])) <= r:
                    total += 1
        return total

    a, b = count(m), count(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(b / a))


def rhythm_features(rr_intervals: np.ndarray) -> dict[str, float]:
    """Compute Group A descriptors from RR intervals in seconds."""
    rr = np.asarray(rr_intervals, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    mean = float(np.mean(rr))
    sd = float(np.std(rr))
    diff = np.diff(rr)

    cv = sd / mean if mean > 1e-12 else 0.0
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0

    # Shannon entropy of a 16-bin normalised histogram.
    hist, _ = np.histogram(rr, bins=16)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))

    sample_entropy = _sample_entropy(rr)
    sd1 = float(np.std(diff) / np.sqrt(2)) if diff.size else 0.0
    sd2 = sd

    if rr.size >= 4 and sd > 1e-12:
        z = rr - rr.mean()
        ac = float(np.dot(z[:-1], z[1:]) / np.dot(z, z))
        periodicity = float(np.clip(ac, -1, 1))
    else:
        periodicity = 1.0 if sd <= 1e-12 else 0.0

    irr = float(np.mean(np.abs(diff)) / mean) if diff.size and mean > 1e-12 else 0.0

    return {
        "rr_cv": float(cv),
        "rr_rmssd": rmssd,
        "rr_entropy": entropy,
        "rr_sample_entropy": sample_entropy,
        "rr_poincare_sd1": sd1,
        "rr_poincare_sd2": sd2,
        "rr_periodicity": periodicity,
        "rr_irregularity_index": irr,
    }
