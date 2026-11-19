"""Stage 3.6.6 — losses and gradient reversal behave correctly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from ecgdet.domain_adaptation import (gradient_reversal, coral_loss, mmd_loss,
                                      domain_adversarial_loss)


def test_gradient_reversal_negates_gradient():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = gradient_reversal(x, alpha=1.0)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-1.0, -1.0]))


def test_coral_and_mmd_zero_for_identical_distributions():
    torch.manual_seed(0)
    a = torch.randn(32, 8)
    b = a.clone()
    assert coral_loss(a, b).item() < 1e-6
    assert mmd_loss(a, b).item() < 1e-6


def test_coral_and_mmd_positive_for_shift():
    torch.manual_seed(1)
    a = torch.randn(64, 8)
    b = a + 3.0
    assert coral_loss(a, b).item() > 0.1
    assert mmd_loss(a, b).item() > 0.01


def test_domain_adversarial_loss_is_differentiable():
    logits = torch.randn(16, 1, requires_grad=True)
    target = torch.ones(16, 1)
    loss = domain_adversarial_loss(logits, target, alpha=0.5)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
