"""Stage 3.6.12 — foundation-model adapter / SSL pretraining tests (TDD)."""
import sys
import pathlib

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.foundation_adapter import (  # noqa: E402
    MaskedSignalEncoder, mask_signal, ssl_reconstruction_loss, ecg_fm_preflight,
)


def test_mask_signal_masks_only_requested_positions():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 500)
    xm, keep = mask_signal(x, mask_prob=0.5, seed=0)
    assert xm.shape == x.shape
    assert keep.shape == x.shape
    assert 0 < keep.float().mean() < 1
    assert torch.all(xm[~keep] == 0)


def test_ssl_loss_ignores_unmasked_positions():
    x = torch.randn(2, 12, 500)
    recon = torch.zeros_like(x) + 5.0
    keep = torch.ones_like(x, dtype=torch.bool)
    keep[:, :, :10] = False  # only the first 10 positions are masked
    loss = ssl_reconstruction_loss(recon, x, keep)
    # Only masked positions contribute; constant 5 vs random x.
    expected = torch.nn.functional.mse_loss(recon[:, :, :10], x[:, :, :10])
    assert torch.allclose(loss, expected, atol=1e-5)


def test_ssl_encoder_reconstructs_shape():
    net = MaskedSignalEncoder(n_leads=12, n_samples=500, dim=16)
    x = torch.randn(2, 12, 500)
    z, recon = net(x)
    assert z.shape == (2, 16)
    assert recon.shape == x.shape


def test_ecg_fm_preflight_reports_missing_backend_without_downloading():
    r = ecg_fm_preflight(checkpoint_dir=None)
    assert "available" in r
    assert isinstance(r["available"], bool)
    assert "reason" in r
