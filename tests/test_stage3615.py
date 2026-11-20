"""Stage 3.6.15 — isolation, source audit, harmonization invariance (TDD)."""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.domain_evaluation import (  # noqa: E402
    split_target_domain, validate_target_isolation,
)
from ecgdet.source_signal_audit import audit_signal  # noqa: E402
from ecgdet.dataset_harmonization import (  # noqa: E402
    canonicalize_leads, harmonize_signal, decimate_signal, HarmonizationStatus,
)


def test_target_adaptation_and_test_are_disjoint():
    meta = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(20)],
        "patient_key": [f"p{i//2}" for i in range(20)],
        "source": ["ptbxl"] * 20,
    })
    a, t = split_target_domain(meta, adaptation_fraction=0.5, seed=42)
    assert validate_target_isolation(a, t, meta) == []
    assert len(a) > 0 and len(t) > 0
    assert set(a).isdisjoint(set(t))


def test_validate_target_isolation_detects_index_and_patient_leak():
    meta = pd.DataFrame({
        "record_id": ["r1", "r2", "r3"],
        "patient_key": ["p1", "p1", "p2"],
    })
    bad = validate_target_isolation([0, 1], [1, 2], meta)
    assert any("index" in v for v in bad)
    assert any("patient" in v for v in bad)


def test_signal_audit_reports_core_quality_fields():
    x = np.zeros((12, 5000), dtype=np.float32)
    x[1] = np.sin(np.arange(5000) / 50)
    r = audit_signal(x, 500.0, [f"L{i}" for i in range(12)])
    for key in ("mean", "std", "rms", "nan_fraction", "inf_fraction",
                "zero_fraction", "baseline_drift", "hf_ratio", "n_leads"):
        assert key in r
    assert r["n_leads"] == 12
    assert r["zero_fraction"] > 0.5


def test_canonicalize_leads_is_permutation_invariant():
    names = ["II", "I", "V1", "V2", "V3", "V4", "V5", "V6", "aVR", "aVL", "aVF", "III"]
    order, mask = canonicalize_leads(names, n_channels=12)
    _, mask2 = canonicalize_leads(list(reversed(names)), n_channels=12)
    assert np.array_equal(mask, mask2)
    assert len(order) == 12


def test_antialias_decimation_suppresses_out_of_band_component():
    fs, target = 500.0, 125.0
    t = np.arange(5000) / fs
    # 200 Hz lies above the 62.5 Hz Nyquist of the target rate.
    x = np.sin(2 * np.pi * 200 * t)
    naive = x[::4]
    clean = decimate_signal(x, fs, target)
    # Naive slicing aliases 200 Hz -> 50 Hz; the filtered version must not.
    def power_near_50(v, f):
        sp = np.abs(np.fft.rfft(v))
        fr = np.fft.rfftfreq(len(v), 1 / f)
        return sp[(fr > 40) & (fr < 60)].sum()
    assert power_near_50(clean, target) < 0.1 * power_near_50(naive, target)


def test_source_classifier_uses_only_signal_features():
    from ecgdet.source_classifier import build_source_features, evaluate_source_classifier
    rng = np.random.default_rng(0)
    X = np.stack([rng.normal(0, 1 + k, (12, 1000)) for k in range(30)])
    f = build_source_features(X, 500.0)
    assert f.shape == (30, 11)
    assert np.isfinite(f).all()
    sources = np.array(["chapman"] * 10 + ["ptbxl"] * 10 + ["cpsc2018_extra"] * 10)
    r = evaluate_source_classifier(f, sources, seed=42)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr", "confusion_matrix"):
        assert k in r


def test_physical_signal_does_not_apply_gain_twice():
    fs = 500.0
    rng = np.random.default_rng(0)
    t = np.arange(5000) / fs
    beat = np.sin(2 * np.pi * 2 * t)
    sig = np.tile(beat, (12, 1)) + rng.normal(0, 0.01, (12, 5000))
    names = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]
    out, _, info = harmonize_signal(sig, fs, names, units=["mV"] * 12,
                                    signal_kind="physical", target_fs=fs, target_len=5000)
    assert info["status"] != HarmonizationStatus.INVALID
    # Physical input must not be multiplied by adc_gain a second time.
    assert np.allclose(out, sig, atol=1e-2)
