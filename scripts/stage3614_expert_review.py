"""Stage 3.6.14 commit 5 — build a blinded cardiologist review package.

Reviewers see ONLY an anonymised 12-lead ECG card.  The manifest that maps
anonymous ids to source/label/model prediction is kept separately and must not
be shared with reviewers.  Actual expert decisions are entered afterwards.
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
D = ROOT / "data" / "processed" / "af_afl_stage3614"
REV = D / "expert_review"
REV.mkdir(parents=True, exist_ok=True)
CARDS = REV / "ecg_cards"
CARDS.mkdir(exist_ok=True)
MET = ROOT / "reports" / "metrics" / "stage3614"
N_PER_STRATUM = 40
N_TARGETED = 30
SEED = 42


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    picks = []
    for src in ("chapman", "ptbxl", "cpsc2018_extra"):
        for lab in (1, 0):
            idx = np.where((meta.source == src) & (meta.label == lab))[0]
            n = min(N_PER_STRATUM, len(idx))
            picks.extend(rng.choice(idx, size=n, replace=False).tolist())
    # Targeted sample: ambiguous / conflicting metadata, analysed separately.
    excluded = pd.read_csv(D / "excluded_records.csv")
    tgt = []
    if len(excluded):
        cand = excluded.index.to_numpy()
        tgt = rng.choice(cand, size=min(N_TARGETED, len(cand)), replace=False).tolist()

    rows = []
    for k, i in enumerate(picks):
        rows.append({
            "anon_id": f"REV{k+1:04d}", "row_idx": int(i),
            "record_id": meta.record_id.iloc[i], "source": meta.source.iloc[i],
            "original_label": int(meta.label.iloc[i]),
            "original_rhythm": meta.rhythm.iloc[i],
            "sample_type": "stratified_random",
        })
    for j, i in enumerate(tgt):
        rows.append({
            "anon_id": f"TGT{j+1:04d}", "row_idx": -1,
            "record_id": excluded.record_id.iloc[i], "source": excluded.source.iloc[i],
            "original_label": -1, "original_rhythm": "AMBIGUOUS",
            "sample_type": "targeted_ambiguous",
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(REV / "review_manifest.csv", index=False)

    cols = ["anon_id", "label", "confidence", "diagnostic_features",
            "transition_rhythm", "signal_quality", "needs_adjudication", "comments"]
    for reviewer in (1, 2):
        t = pd.DataFrame(columns=cols)
        t["anon_id"] = manifest.anon_id
        t.to_csv(REV / f"reviewer_{reviewer}.csv", index=False)
    pd.DataFrame(columns=["anon_id", "reviewer_1_label", "reviewer_2_label",
                          "adjudicated_label", "adjudicator", "notes"]).assign(
        anon_id=manifest.anon_id).to_csv(REV / "adjudication.csv", index=False)
    pd.DataFrame(columns=["anon_id", "final_expert_label", "confidence"]).assign(
        anon_id=manifest.anon_id).to_csv(REV / "final_expert_labels.csv", index=False)

    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    t = np.arange(5000) / 500.0
    for r in manifest[manifest.row_idx >= 0].itertuples():
        sig = np.asarray(X[r.row_idx], dtype=np.float32)
        fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)
        for k, ax in enumerate(axes):
            ax.plot(t, sig[k], lw=0.6, color="black")
            ax.set_ylabel(leads[k], rotation=0, labelpad=12, fontsize=8)
            ax.set_yticks([])
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"ECG {r.anon_id}", fontsize=11)
        fig.subplots_adjust(hspace=0, top=0.97)
        fig.savefig(CARDS / f"{r.anon_id}.png", dpi=90)
        plt.close(fig)

    stats = {
        "n_stratified": int((manifest.sample_type == "stratified_random").sum()),
        "n_targeted": int((manifest.sample_type == "targeted_ambiguous").sum()),
        "by_source_label": {f"{s}|{l}": int(n) for (s, l), n in
                            manifest[manifest.sample_type == "stratified_random"]
                            .groupby(["source", "original_label"]).size().items()},
        "seed": SEED,
        "blinding": "reviewers receive only anon_id and ECG card; no source/label/prediction",
        "note": "Expert decisions are not filled in — this package is a protocol template.",
    }
    (MET / "expert_review_package.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
