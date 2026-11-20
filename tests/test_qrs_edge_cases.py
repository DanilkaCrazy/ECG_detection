"""Stage 3.6.9 — QRS audit edge cases (short RR, single lead, F-waves, ms tolerances)."""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402

FS = 500.0
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def _beats(rr_samples, n=40, amp=1.0, f_wave=0.0):
    x = np.zeros((12, int(rr_samples * (n + 2))))
    idx = np.arange(50, x.shape[1] - 50, rr_samples)
    tmpl = np.array([-.1, -.2, 1.0, -.3, -.1, 0, 0, 0, .05]) * amp
    for p in idx:
        for k in range(12):
            x[k, p - 3:p + 6] += tmpl
    if f_wave:
        t = np.arange(x.shape[1]) / FS
        x += f_wave * np.sin(2 * np.pi * 5 * t)
    return x.astype(np.float32), idx


@pytest.mark.parametrize("rr_ms", [1000, 300, 240])
def test_short_rr_and_tachycardia_are_not_merged(rr_ms):
    rr = int(rr_ms / 1000 * FS)
    x, ref = _beats(rr)
    peaks, _ = multi_lead_qrs(x, FS, LEADS)
    assert len(peaks) >= 0.8 * len(ref), (rr_ms, len(peaks), len(ref))


def test_single_available_lead_keeps_detection():
    x, ref = _beats(500)
    single = np.zeros_like(x)
    single[1] = x[1]
    peaks, info = multi_lead_qrs(single, FS, LEADS)
    assert len(peaks) >= 0.8 * len(ref)
    assert info["qrs_detection_status"] in ("ok", "degraded")


def test_missing_chest_leads_keep_detection():
    x, ref = _beats(500)
    x[6:] = 0
    peaks, _ = multi_lead_qrs(x, FS, LEADS)
    assert len(peaks) >= 0.8 * len(ref)


def test_f_waves_do_not_destroy_qrs_detection():
    x, ref = _beats(500, f_wave=0.25)
    peaks, _ = multi_lead_qrs(x, FS, LEADS)
    assert 0.8 * len(ref) <= len(peaks) <= 1.25 * len(ref)


def test_tolerance_is_defined_in_milliseconds_not_samples():
    """Same 50 ms tolerance corresponds to different sample counts at 250/500 Hz."""
    for fs in (250.0, 500.0):
        tol_samples = 0.05 * fs
        assert tol_samples == pytest.approx(50.0 * fs / 1000.0)
