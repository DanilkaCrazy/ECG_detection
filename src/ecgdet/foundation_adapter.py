"""Stage 3.6.12 — ECG foundation-model adapter with an honest SSL fallback.

ECG-FM (wanglab/ecg-fm, MIT) is a 24-layer wav2vec2-style model distributed as
a fairseq checkpoint (~1.09 GB).  Loading it requires the ``fairseq`` backend,
which is not part of this environment.  ``ecg_fm_preflight`` therefore reports
the blocker explicitly instead of silently substituting another model.

The fallback implemented here is a *local* self-supervised masked-signal
encoder trained on the TRAIN split only.  It is NOT ECG-FM transfer and must be
reported as such.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "ecg_fm_preflight", "mask_signal", "ssl_reconstruction_loss",
    "MaskedSignalEncoder",
]


def ecg_fm_preflight(checkpoint_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Check whether real ECG-FM weights can be used here; never downloads."""
    fairseq = importlib.util.find_spec("fairseq") is not None
    ckpt = None
    if checkpoint_dir is not None:
        cand = pathlib.Path(checkpoint_dir)
        for name in ("mimic_iv_ecg_physionet_pretrained.pt", "mimic_iv_ecg_finetuned.pt"):
            if (cand / name).exists():
                ckpt = str(cand / name)
                break
    reasons = []
    if not fairseq:
        reasons.append("fairseq backend is not installed (ECG-FM checkpoint is a fairseq model)")
    if ckpt is None:
        reasons.append("no local ECG-FM checkpoint found")
    return {
        "available": fairseq and ckpt is not None,
        "fairseq_installed": fairseq,
        "checkpoint": ckpt,
        "reason": "; ".join(reasons) if reasons else "ready",
        "note": "When unavailable, Stage 3.6.12 trains a LOCAL SSL encoder — not ECG-FM.",
    }


def mask_signal(x: torch.Tensor, mask_prob: float = 0.4,
                seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly zero time points; returns (masked_x, keep_mask)."""
    if seed is None:
        keep = torch.rand_like(x) > mask_prob
    else:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        keep = torch.rand(x.shape, generator=gen, dtype=torch.float32) > mask_prob
        keep = keep.to(x.device)
    return x * keep, keep


def ssl_reconstruction_loss(recon: torch.Tensor, x: torch.Tensor,
                            keep: torch.Tensor) -> torch.Tensor:
    """MSE computed ONLY on masked positions (keep=False)."""
    missing = ~keep
    if missing.sum() == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    return torch.nn.functional.mse_loss(recon[missing], x[missing])


class MaskedSignalEncoder(nn.Module):
    """Small 1D convolutional encoder + decoder for masked-signal modelling."""

    def __init__(self, n_leads: int = 12, n_samples: int = 1250, dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_leads, 16, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 32 * 8), nn.Unflatten(1, (32, 8)),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=3, padding=7),
            nn.ReLU(),
            nn.ConvTranspose1d(16, n_leads, kernel_size=25, stride=5, padding=12,
                               output_padding=0),
            nn.AdaptiveAvgPool1d(n_samples),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)
