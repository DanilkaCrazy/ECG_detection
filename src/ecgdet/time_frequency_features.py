"""Stage 3.6.10 — Group C: time-frequency descriptors (existing STFT array).

The 16-column ``features_temporal_organization.npy`` is split here: STFT
descriptors form group C, while ``org_*`` organization descriptors belong to
group D.  This keeps each ablation group disjoint and interpretable.
"""
from __future__ import annotations

__all__ = ["select_tf_features", "select_organization_features"]


def select_tf_features(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith("stft_") or n.startswith("tf_")]


def select_organization_features(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith("org_")]
