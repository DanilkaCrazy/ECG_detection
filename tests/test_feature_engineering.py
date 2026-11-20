"""Stage 3.6.10 — registry, selection and morphology unit tests."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.feature_registry import select_group_columns  # noqa: E402
from ecgdet.feature_selection import prune_constants, correlation_prune, select_features  # noqa: E402
from ecgdet.morphology_features import morphology_features  # noqa: E402


def test_registry_assigns_each_feature_to_exactly_one_group():
    names = ["rr_cv", "adaptive_II_dominant_freq", "stft_tf_entropy_mean",
             "org_periodicity", "interlead_plv_mean", "diagnostic_qrs_supp"]
    g = select_group_columns(names)
    assert g["A"] == [0]
    assert g["B"] == [1]
    assert g["C"] == [2]
    assert g["D"] == [3]
    assert g["E"] == [4]
    assert sum(len(v) for v in g.values()) == 5


def test_prune_constants_drops_constant_and_nan_columns():
    X = np.array([[1, 5, np.nan], [2, 5, np.nan], [3, 5, np.nan]], float)
    Xp, names, idx = prune_constants(X, ["a", "b", "c"])
    assert names == ["a"]
    assert idx == [0]


def test_correlation_prune_removes_duplicate_columns():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    X = np.column_stack([a, a, rng.normal(size=100)])
    _, names, idx = correlation_prune(X, ["a", "a_dup", "b"])
    assert names == ["a", "b"]
    assert idx == [0, 2]


def test_select_features_uses_only_fit_data(monkeypatch):
    from ecgdet import feature_selection as fs
    seen = {}

    def fake_fit(self, X, y):
        seen["shape"] = X.shape
        return self

    monkeypatch.setattr(fs.SelectKBest, "fit", fake_fit)
    monkeypatch.setattr(fs.SelectKBest, "get_support", lambda self: np.array([True, False]))
    idx, names = select_features(np.ones((10, 2)), np.array([0, 1] * 5), ["x", "y"], k=1)
    assert seen["shape"] == (10, 2)
    assert idx == [0] and names == ["x"]


def test_morphology_reproducibility_is_high_for_identical_beats():
    fs = 500.0
    t = np.arange(5000) / fs
    sig = np.zeros(5000)
    for p in np.arange(250, 4800, 500):
        sig[p:p + 40] = np.hanning(40)
    peaks = np.arange(250, 4800, 500)
    f = morphology_features(sig, peaks, fs)
    assert f["morph_cycle_corr"] > 0.9
    assert f["morph_amplitude_cv"] < 0.05
    assert f["morph_template_similarity"] > 0.9


def test_morphology_handles_too_few_beats():
    f = morphology_features(np.zeros(100), np.array([50]), 500.0)
    assert all(v == 0.0 for v in f.values())
