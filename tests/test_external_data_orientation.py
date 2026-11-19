"""Stage 3.6 P0 — regression test for WFDB (samples, channels) orientation."""
import sys, pathlib, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import build_external_af_dataset as b


def _fake_record():
    """WFDB-like object: p_signal is (samples, channels)."""
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 0.1, (5000, 12))
    return types.SimpleNamespace(
        p_signal=sig, fs=500.0,
        sig_name=["I", "II", "III", "AVR", "AVL", "AVF"] + [f"V{i}" for i in range(1, 7)],
        units=["mV"] * 12,
    )


def test_load_one_accepts_wfdb_sample_major_signal(monkeypatch):
    import wfdb
    monkeypatch.setattr(wfdb, "rdrecord", lambda _p: _fake_record())
    out = b.load_one({"path": "fake"})
    assert out is not None, "record was rejected — orientation fix failed"
    assert out["signal"].shape == (12, 5000)
    assert np.mean(np.abs(out["signal"]) > 1e-8) > 0.5


def test_load_one_rejects_mostly_zero_padded_signal(monkeypatch):
    import wfdb
    rec = _fake_record()
    rec.p_signal = np.zeros_like(rec.p_signal)
    rec.p_signal[:12] = 1.0  # the old zero-padded failure mode
    monkeypatch.setattr(wfdb, "rdrecord", lambda _p: rec)
    assert b.load_one({"path": "fake"}) is None
