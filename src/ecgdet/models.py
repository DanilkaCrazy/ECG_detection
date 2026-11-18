"""Reusable 1D CNN / ResNet raw-ECG architectures for the arrhythmia task."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "CNN1D",
    "ResNet1D",
    "BasicBlock1D",
    "build_model",
]


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CNN1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 12, dropout: float = 0.3, binary: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        out_dim = 1 if binary else num_classes
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 12, dropout: float = 0.3, binary: bool = False):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        out_dim = 1 if binary else num_classes
        self.fc = nn.Linear(128, out_dim)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(BasicBlock1D(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            else:
                layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(
    name: str,
    num_classes: int,
    in_channels: int = 12,
    dropout: float = 0.3,
    binary: bool = False,
) -> nn.Module:
    """Build a raw-ECG model.

    ``binary=True`` replaces the softmax head with a single sigmoid logit
    (used by the AF-vs-AFL specialised head in the cascade).
    """
    if name == "CNN1D":
        return CNN1D(num_classes, in_channels=in_channels, dropout=dropout, binary=binary)
    if name == "ResNet1D":
        return ResNet1D(num_classes, in_channels=in_channels, dropout=dropout, binary=binary)
    raise ValueError(f"unknown model {name!r}")
