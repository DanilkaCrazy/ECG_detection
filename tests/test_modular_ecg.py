"""Stage 3.6.11 — modular ECG architecture tests (TDD)."""
import sys
import pathlib

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.modular_ecg import (  # noqa: E402
    AttentionFusion, LeadQualityGate, ModularECGNet, modular_loss,
)


def test_lead_quality_gate_masks_missing_leads():
    gate = LeadQualityGate()
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    w = gate(mask)
    assert w.shape == (1, 12)
    assert torch.all(w[0, 2:] == 0)
    assert torch.all(w[0, :2] > 0)


def test_attention_fusion_returns_normalised_module_weights():
    fusion = AttentionFusion(n_modules=3, dim=16)
    toks = torch.randn(5, 3, 16)
    fused, w = fusion(toks)
    assert fused.shape == (5, 16)
    assert w.shape == (5, 3)
    assert torch.allclose(w.sum(dim=1), torch.ones(5), atol=1e-5)


def test_modular_net_forward_and_aux_outputs():
    torch.manual_seed(0)
    net = ModularECGNet(n_leads=12, n_samples=5000, n_rr=40)
    x = torch.randn(2, 12, 5000)
    rr = torch.rand(2, 40) * 0.5 + 0.4
    mask = torch.ones(2, 12, dtype=torch.bool)
    logits, aux = net(x, rr, mask)
    assert logits.shape == (2,)
    for key in ("module_weights", "quality", "p_afib", "p_afl"):
        assert key in aux
    assert aux["module_weights"].shape == (2, 3)
    assert torch.allclose(aux["p_afib"] + aux["p_afl"], torch.ones(2), atol=1e-5)


def test_modular_loss_is_finite_without_atrial_target():
    net = ModularECGNet(n_leads=12, n_samples=5000, n_rr=40)
    x = torch.randn(2, 12, 5000)
    rr = torch.rand(2, 40) + 0.4
    mask = torch.ones(2, 12, dtype=torch.bool)
    y = torch.tensor([0.0, 1.0])
    logits, aux = net(x, rr, mask)
    loss, parts = modular_loss(logits, y, aux)
    assert torch.isfinite(loss)
    assert "bce" in parts
