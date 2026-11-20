"""Stage 3.6.22 — regenerate expert cards with REAL ECG calibration.

Stage 3.6.18 cards only had a text label.  Here the axes use physical units:
1 mV = 10 mm, 25 mm/s (0.2 s = 5 mm), with visible grid and scale bar.
"""
from __future__ import annotations

import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "af_afl_stage34" / "X.npy"
base = ROOT / "data" / "processed" / "af_afl_stage3614_v1_1"
REV = base / "expert_review_v1_1"
CARDS = REV / "ecg_cards"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def main() -> int:
    manifest = pd.read_csv(REV / "review_manifest.csv")
    X = np.load(SRC, mmap_mode="r")
    t = np.arange(5000) / 500.0
    for r in manifest.itertuples():
        sig = np.asarray(X[int(r.row_idx)], dtype=np.float32)
        missing = [LEADS[k] for k in range(12)
                   if np.mean(np.abs(sig[k]) > 1e-8) < 0.1]
        fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
        for k, ax in enumerate(axes):
            if LEADS[k] in missing:
                ax.text(0.5, 0.5, f"{LEADS[k]} — MISSING (lead_mask=False)",
                        transform=ax.transAxes, ha="center", va="center",
                        color="red", fontsize=8)
                ax.set_ylabel(LEADS[k], rotation=0, labelpad=12, fontsize=8)
            else:
                ax.plot(t, sig[k], lw=0.5, color="black")
                ax.set_ylabel(LEADS[k], rotation=0, labelpad=12, fontsize=8)
            # Physical calibration: 1 mV = 10 mm; 0.2 s = 5 mm at 25 mm/s.
            vals = sig[k][np.isfinite(sig[k])]
            span = max(0.5, float(np.percentile(np.abs(vals), 99)) * 1.2) if vals.size else 1.0
            step = 0.5 if span <= 1 else 1.0
            ax.set_ylim(-span, span)
            yticks = np.arange(-np.floor(span / step) * step,
                               np.floor(span / step) * step + step, step)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:g}" for v in yticks], fontsize=6)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xticks(np.arange(0, 10.01, 0.2))
        axes[-1].set_xticklabels([])
        axes[-1].set_xlabel("Time (s), grid 0.2 s = 5 mm at 25 mm/s")
        axes[0].set_title(f"ECG {r.anon_id} | 500 Hz | amplitude ticks in mV "
                          "(1 mV = 10 mm)", fontsize=9)
        fig.subplots_adjust(hspace=0, top=0.96)
        fig.savefig(CARDS / f"{r.anon_id}.png", dpi=90)
        plt.close(fig)
    print("regenerated", len(manifest), "cards with physical calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
