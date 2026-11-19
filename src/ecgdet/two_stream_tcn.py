"""Stage 3.6.7 — two-stream TCN prototype (QRST vs atrial separation).

Multi-task objective:
  L = w_rec * L_reconstruction + w_atr * L_atrial + w_cls * L_classification
L_atrial is ONLY used on synthetic signals with a known atrial component.
This is a research prototype, not a clinically validated model.
"""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["TCNBlock", "TwoStreamTCN", "multitask_loss"]


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU())
        self.res = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        return self.net(x) + self.res(x)


class TwoStreamTCN(nn.Module):
    def __init__(self, in_channels=12, length=5000, base=32, binary=True):
        super().__init__()
        self.length = length
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 7, stride=2, padding=3), nn.BatchNorm1d(base), nn.ReLU())
        self.tcn = nn.Sequential(TCNBlock(base, dilation=1), TCNBlock(base, dilation=2),
                                 TCNBlock(base, dilation=4))
        self.qrst_dec = nn.Sequential(nn.ConvTranspose1d(base, base, 4, stride=2, padding=1),
                                      nn.Conv1d(base, in_channels, 7, padding=3))
        self.atrial_dec = nn.Sequential(nn.ConvTranspose1d(base, base, 4, stride=2, padding=1),
                                        nn.Conv1d(base, in_channels, 7, padding=3))
        self.atrial_proj = nn.Conv1d(in_channels, base, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Linear(base, 1 if binary else 2)

    def forward(self, x):
        z = self.tcn(self.stem(x))
        qrst = self.qrst_dec(z)
        atrial = self.atrial_dec(z)
        if qrst.shape[-1] != x.shape[-1]:
            qrst = nn.functional.interpolate(qrst, size=x.shape[-1], mode="linear", align_corners=False)
            atrial = nn.functional.interpolate(atrial, size=x.shape[-1], mode="linear", align_corners=False)
        feat = self.pool(self.atrial_proj(atrial)).squeeze(-1)
        logit = self.cls(feat).squeeze(-1)
        return qrst, atrial, logit


def multitask_loss(x, qrst_hat, atrial_hat, y_logit, y_true,
                   atrial_true=None, w_rec=1.0, w_atr=1.0, w_cls=1.0):
    rec = nn.functional.mse_loss(qrst_hat + atrial_hat, x)
    cls = nn.functional.binary_cross_entropy_with_logits(y_logit, y_true)
    if atrial_true is None:
        return w_rec * rec + w_cls * cls, {"rec": rec.item(), "cls": cls.item(), "atr": None}
    atr = nn.functional.mse_loss(atrial_hat, atrial_true)
    return (w_rec * rec + w_atr * atr + w_cls * cls,
            {"rec": rec.item(), "cls": cls.item(), "atr": atr.item()})
