"""Stage 3.6.6 — domain-invariant learning primitives (DANN, CORAL, MMD)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["GradientReversal", "gradient_reversal", "coral_loss", "mmd_loss",
           "domain_adversarial_loss"]


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


def coral_loss(source, target):
    """CORAL: match second-order statistics of source and target features."""
    if source.size(0) < 2 or target.size(0) < 2:
        return torch.tensor(0.0, device=source.device, requires_grad=True)
    ns, nt = source.size(0), target.size(0)
    d = source.size(1)
    cs = torch.cov(source.T)
    ct = torch.cov(target.T)
    ms = source.mean(0)
    mt = target.mean(0)
    return ((cs - ct) ** 2).sum() / (4 * d * d) + (ms - mt).pow(2).mean()


def _rbf_kernel(x, y, sigma):
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd_loss(source, target, sigmas=(1.0, 2.0, 4.0, 8.0)):
    """Multi-kernel MMD^2 between source and target feature batches."""
    loss = torch.tensor(0.0, device=source.device)
    for s in sigmas:
        kxx = _rbf_kernel(source, source, s)
        kyy = _rbf_kernel(target, target, s)
        kxy = _rbf_kernel(source, target, s)
        loss = loss + kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return loss


def domain_adversarial_loss(domain_logits, domain_target, alpha=1.0):
    """BCE on reversed-gradient domain logits (DANN)."""
    rev = gradient_reversal(domain_logits, alpha)
    return nn.functional.binary_cross_entropy_with_logits(rev, domain_target)
