"""Stage 3.6.x — reliability metrics (ECE) for probability calibration."""
import numpy as np


def expected_calibration_error(y, p, n_bins=10):
    """ECE: |confidence - accuracy| averaged over equal-width confidence bins."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += float(m.mean() * abs(float(y[m].mean()) - float(p[m].mean())))
    return ece
