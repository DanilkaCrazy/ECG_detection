"""Stage 3.6.11 — modular, interpretable AFIB/AFL network.

Three independent encoders follow the modular design of Fleury et al. (2024),
adapted to 10-s 12-lead ECG:

    RhythmEncoder     <- RR interval sequence (ventricular regularity)
    AtrialEncoder     <- QRST-cancelled residual (atrial activity)
    MorphologyEncoder <- raw 12-lead ECG (QRS/T morphology)

A LeadQualityGate turns the availability mask into per-lead weights so that an
absent lead is never interpreted as a physiologically meaningful zero signal.
An attention Fusion produces a fused representation AND per-module weights,
which are saved as an interpretability artifact.

Important limitation: the atrial encoder consumes an externally-cancelled
residual; the network does NOT learn to separate QRST from atrial activity
end-to-end.  Module weights show which *evidence stream* the classifier used,
not a physiological decomposition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LeadQualityGate", "RhythmEncoder", "SignalEncoder", "AttentionFusion",
    "ModularECGNet", "modular_loss",
]


class LeadQualityGate(nn.Module):
    """Normalise a boolean lead mask into non-negative per-lead weights."""

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(torch.float32)
        return m / m.sum(dim=1, keepdim=True).clamp(min=1.0)


class RhythmEncoder(nn.Module):
    """Encode an RR-interval sequence into a fixed-size vector."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.stats = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16 * 8 + 16, dim), nn.ReLU())

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        # rr: (B, N) in seconds, zeros denote padding.
        valid = (rr > 0).float()
        n = valid.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (rr * valid).sum(1, keepdim=True) / n
        sd = torch.sqrt(((rr - mean) ** 2 * valid).sum(1, keepdim=True) / n)
        diff = torch.abs(rr[:, 1:] - rr[:, :-1]) * valid[:, 1:]
        rmssd = torch.sqrt(diff.pow(2).sum(1, keepdim=True) / valid[:, 1:].sum(1, keepdim=True).clamp(min=1))
        stats = torch.cat([mean, sd, rmssd, n / rr.shape[1]], dim=1)
        seq = self.conv(rr.unsqueeze(1)).flatten(1)
        return self.head(torch.cat([seq, self.stats(stats)], dim=1))


class SignalEncoder(nn.Module):
    """Compact 1D CNN for a (leads, samples) signal."""

    def __init__(self, n_leads: int, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, 12, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(12), nn.ReLU(),
            nn.Conv1d(12, 16, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(nn.Linear(16 * 16, dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


class AttentionFusion(nn.Module):
    """Attention over module tokens with a learned query."""

    def __init__(self, n_modules: int, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.key = nn.Linear(dim, dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, M, D)
        scores = torch.matmul(self.key(tokens), self.query.transpose(1, 2)).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        fused = torch.einsum("bm,bmd->bd", weights, tokens)
        return fused, weights


class ModularECGNet(nn.Module):
    """Three-stream ECG classifier with attention fusion and audit outputs."""

    def __init__(self, n_leads: int = 12, n_samples: int = 5000, n_rr: int = 40,
                 dim: int = 32):
        super().__init__()
        self.gate = LeadQualityGate()
        self.rhythm = RhythmEncoder(dim)
        self.atrial = SignalEncoder(n_leads, dim)
        self.morphology = SignalEncoder(n_leads, dim)
        self.fusion = AttentionFusion(n_modules=3, dim=dim)
        self.cls = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.quality = nn.Sequential(nn.Linear(n_leads, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, rr: torch.Tensor, mask: torch.Tensor):
        w = self.gate(mask)
        xw = x * w.unsqueeze(-1)
        toks = torch.stack([
            self.rhythm(rr),
            self.atrial(xw),
            self.morphology(x),
        ], dim=1)
        fused, weights = self.fusion(toks)
        logits = self.cls(fused).squeeze(-1)
        p = torch.sigmoid(logits)
        aux = {
            "module_weights": weights,
            "quality": self.quality(w),
            "p_afib": p,
            "p_afl": 1 - p,
        }
        return logits, aux


def modular_loss(logits: torch.Tensor, y: torch.Tensor, aux: dict,
                 entropy_weight: float = 0.01) -> tuple[torch.Tensor, dict]:
    """BCE + small entropy penalty that discourages degenerate fusion."""
    n_pos = max(int((y == 1).sum().item()), 1)
    pos = float(int((y == 0).sum().item()) / n_pos)
    bce = F.binary_cross_entropy_with_logits(logits, y,
                                             pos_weight=torch.tensor([pos], device=logits.device))
    w = aux["module_weights"]
    entropy = -(w * w.clamp_min(1e-8).log()).sum(dim=1).mean()
    loss = bce - entropy_weight * entropy
    return loss, {"bce": float(bce.detach()), "entropy": float(entropy.detach())}
