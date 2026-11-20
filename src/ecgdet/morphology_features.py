"""Stage 3.6.10 — Group D: morphology / cycle-reproducibility descriptors."""
from __future__ import annotations

import numpy as np

__all__ = ["morphology_features", "FEATURE_NAMES"]

FEATURE_NAMES = (
    "morph_cycle_corr", "morph_amplitude_cv", "morph_template_similarity",
)


def morphology_features(signal: np.ndarray, peaks: np.ndarray, fs: float,
                        half_ms: float = 120.0) -> dict[str, float]:
    """Describe reproducibility of successive beat-aligned waveforms."""
    x = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    half = max(4, int(half_ms / 1000.0 * fs))
    beats = []
    for p in peaks:
        a, b = p - half, p + half
        if a >= 0 and b <= len(x):
            seg = x[a:b]
            beats.append(seg / (np.std(seg) + 1e-9))
    if len(beats) < 3:
        return {k: 0.0 for k in FEATURE_NAMES}

    M = np.asarray(beats)
    corr = np.corrcoef(M)
    iu = np.triu_indices(len(M), k=1)
    mean_corr = float(np.mean(corr[iu])) if iu[0].size else 0.0

    raw = [x[p - half:p + half] for p in peaks if p - half >= 0 and p + half <= len(x)]
    amps = np.array([np.std(seg) for seg in raw])
    amp_cv = float(amps.std() / (amps.mean() + 1e-12))

    template = np.mean(M, axis=0)
    similarity = float(np.mean([np.corrcoef(seg, template)[0, 1] for seg in M]))

    return {
        "morph_cycle_corr": mean_corr,
        "morph_amplitude_cv": amp_cv,
        "morph_template_similarity": similarity,
    }
