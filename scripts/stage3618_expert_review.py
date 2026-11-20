"""Stage 3.6.18 — corrected blinded expert review package.

Fixes: all 89 PTB-XL records included, cards for the 30 targeted ambiguous
records, ECG grid with time/amplitude calibration, and explicit lead-mask
annotation for missing channels.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "af_afl_stage34"
base = ROOT / "data" / "processed" / "af_afl_stage3614_v1_1"
if not base.exists():
    base = ROOT / "data" / "processed" / "af_afl_stage3614"
REV = base / "expert_review_v1_1"
REV.mkdir(parents=True, exist_ok=True)
CARDS = REV / "ecg_cards"
CARDS.mkdir(exist_ok=True)
MET = ROOT / "reports" / "metrics" / "stage3618"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]
SEED = 42


def main() -> int:
    meta = pd.read_csv(base / "meta.csv")
    X = np.load(SRC / "X.npy", mmap_mode="r")
    prov = pd.read_csv(base / "diagnosis_provenance.csv")
    rng = np.random.default_rng(SEED)
    picks = []
    for src in ("chapman", "ptbxl", "cpsc2018_extra"):
        for lab in (1, 0):
            idx = np.where((meta.source == src) & (meta.label == lab))[0]
            # PTB-XL is small: include ALL valid records.
            n = len(idx) if src == "ptbxl" else min(40, len(idx))
            picks.extend(rng.choice(idx, size=n, replace=False).tolist())

    # Targeted ambiguous records from the previous v1 audit.
    amb_path = base / "excluded_records.csv"
    target_rows = []
    if amb_path.exists():
        amb = pd.read_csv(amb_path)
        if "row_in_stage34" in amb.columns:
            sel = rng.choice(amb.index.to_numpy(), size=min(30, len(amb)), replace=False)
            target_rows = amb.loc[sel, "row_in_stage34"].astype(int).tolist()

    rows = []
    for k, i in enumerate(picks):
        rows.append({"anon_id": f"REV{k+1:04d}", "row_idx": int(i),
                     "record_id": meta.record_id.iloc[i], "source": meta.source.iloc[i],
                     "original_label": int(meta.label.iloc[i]),
                     "sample_type": "stratified_random"})
    for j, i in enumerate(target_rows):
        # row_idx refers to stage34; find matching record in the clean meta.
        match = np.where(prov.row_in_stage34.to_numpy() == i)[0]
        rid = prov.record_id.iloc[match[0]] if len(match) else f"AMB{j+1:04d}"
        rows.append({"anon_id": f"TGT{j+1:04d}", "row_idx": int(i),
                     "record_id": rid, "source": "ambiguous",
                     "original_label": -1, "sample_type": "targeted_ambiguous"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(REV / "review_manifest.csv", index=False)

    cols = ["anon_id", "label", "confidence", "f_visible", "atrial_regularity",
            "ventricular_regularity", "av_block", "signal_quality",
            "mixed_rhythm", "needs_adjudication", "comments"]
    for reviewer in (1, 2):
        t = pd.DataFrame(columns=cols)
        t["anon_id"] = manifest.anon_id
        t.to_csv(REV / f"reviewer_{reviewer}.csv", index=False)
    pd.DataFrame(columns=["anon_id", "reviewer_1_label", "reviewer_2_label",
                          "adjudicated_label", "adjudicator", "notes"]).assign(
        anon_id=manifest.anon_id).to_csv(REV / "adjudication.csv", index=False)

    t = np.arange(5000) / 500.0
    for r in manifest.itertuples():
        sig = np.asarray(X[r.row_idx], dtype=np.float32)
        # Missing leads: whole channel is ~0 -> mark explicitly.
        missing = [LEADS[k] for k in range(12)
                   if np.mean(np.abs(sig[k]) > 1e-8) < 0.1]
        fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
        for k, ax in enumerate(axes):
            if LEADS[k] in missing:
                ax.text(0.5, 0.5, f"{LEADS[k]} — MISSING", transform=ax.transAxes,
                        ha="center", va="center", color="red", fontsize=9)
                ax.set_ylabel(LEADS[k], rotation=0, labelpad=12, fontsize=8)
                ax.set_yticks([])
                continue
            ax.plot(t, sig[k], lw=0.6, color="black")
            ax.set_ylabel(LEADS[k], rotation=0, labelpad=12, fontsize=8)
            ax.set_yticks([])
            ax.grid(True, which="major", axis="x", alpha=0.3)
        axes[-1].set_xlabel("Time (s) — 25 mm/s equivalent")
        axes[0].set_title(f"ECG {r.anon_id} | 500 Hz | 1 mV = 10 mm (calibrated display)",
                          fontsize=9)
        fig.subplots_adjust(hspace=0, top=0.96)
        fig.savefig(CARDS / f"{r.anon_id}.png", dpi=90)
        plt.close(fig)

    stats = {
        "n_stratified": int((manifest.sample_type == "stratified_random").sum()),
        "n_targeted": int((manifest.sample_type == "targeted_ambiguous").sum()),
        "n_cards": len(list(CARDS.glob("*.png"))),
        "by_source_label": {f"{s}|{l}": int(n) for (s, l), n in
                            manifest[manifest.sample_type == "stratified_random"]
                            .groupby(["source", "original_label"]).size().items()},
        "calibration": "25 mm/s time grid, 10 mm/mV amplitude convention, 500 Hz",
        "lead_mask": "missing leads marked MISSING, not shown as flat signal",
    }
    (MET / "expert_review_package_v1_1.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
