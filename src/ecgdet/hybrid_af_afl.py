"""Hybrid AF/AFL network: raw-signal CNN branches + physiological feature branches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SEBlock1D", "ResNet1DSE", "RawCNN", "InceptionTime1D", "HybridAFAFL", "FocalLoss1D"]


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation channel attention for 1D signals."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class _SEBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.se = SEBlock1D(cout)
        self.relu = nn.ReLU(inplace=True)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(cout))

    def forward(self, x):
        s = self.skip(x) if self.skip is not None else x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return self.relu(x + s)


class ResNet1DSE(nn.Module):
    """1D ResNet with SE attention for raw 12-lead ECG."""

    def __init__(self, in_channels: int = 12, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_SEBlock(32, 32), _SEBlock(32, 32))
        self.layer2 = nn.Sequential(_SEBlock(32, 64, 2), _SEBlock(64, 64))
        self.layer3 = nn.Sequential(_SEBlock(64, 128, 2), _SEBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, embed_dim))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.head(self.pool(x))


class RawCNN(nn.Module):
    """ResNet1DSE backbone with a binary sigmoid-logit head (1 output)."""

    def __init__(self, in_channels: int = 12, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone = ResNet1DSE(in_channels, embed_dim, dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


class InceptionTime1D(nn.Module):
    """Multi-scale 1D convolutions (different kernel widths in parallel)."""

    def __init__(self, in_channels: int = 12, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        ks = [7, 15, 31, 63]
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels, 32, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                          nn.AdaptiveAvgPool1d(1), nn.Flatten())
            for k in ks
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(32 * len(ks) + in_channels, embed_dim),
                                nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        feats.append(self.pool(x))
        return self.fc(torch.cat(feats, dim=1))


class HybridAFAFL(nn.Module):
    """Fusion of raw morphology, multi-scale patterns, RR sequence and expert features."""

    def __init__(self, n_expert: int = 67, rr_len: int = 20, embed_dim: int = 128,
                 in_channels: int = 12, dropout: float = 0.4):
        super().__init__()
        self.resnet = ResNet1DSE(in_channels, embed_dim, dropout)
        self.inception = InceptionTime1D(in_channels, embed_dim, dropout)
        self.rr_mlp = nn.Sequential(
            nn.Linear(rr_len, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, embed_dim), nn.ReLU(inplace=True))
        self.expert_mlp = nn.Sequential(
            nn.Linear(n_expert, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, embed_dim), nn.ReLU(inplace=True))
        self.attn = nn.Sequential(nn.Linear(embed_dim * 4, 4), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout / 2),
            nn.Linear(64, 1))

    def forward(self, ecg, rr, expert):
        a = self.resnet(ecg)
        b = self.inception(ecg)
        c = self.rr_mlp(rr)
        d = self.expert_mlp(expert)
        stack = torch.stack([a, b, c, d], dim=1)  # (B,4,E)
        w = self.attn(torch.cat([a, b, c, d], dim=1)).unsqueeze(-1)  # (B,4,1)
        fused = (stack * w).sum(dim=1)
        return self.classifier(fused)


class FocalLoss1D(nn.Module):
    """Binary focal loss with class alpha and logits input."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        targets = targets.view(-1, 1).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()
