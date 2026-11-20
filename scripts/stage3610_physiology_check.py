"""Stage 3.6.10 — statistical direction check for top physiologically-interpretable features.

The hypothesis (AFL = organised F-waves, AFIB = disorganised activity) must be
tested against the data, not assumed because SHAP ranks a feature highly.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3610"


def main() -> int:
    fi = pd.read_csv(MET / "feature_importance.csv")
    meta = pd.read_csv(ROOT / "data" / "processed" / "af_afl_stage34" / "meta.csv")
    y = meta.label.to_numpy()
    top = fi.head(12)
    rows = []
    for _, r in top.iterrows():
        name = str(r["feature"])
        if name.startswith("B_"):
            names = pd.read_csv(ROOT / "data" / "processed" / "af_afl_stage3610" / "feature_names.csv")
            idx = np.where(names.name.to_numpy() == name)[0]
            X = np.load(ROOT / "data" / "processed" / "af_afl_stage3610" / "X.npy", mmap_mode="r")
            vals = np.asarray(X[:, idx[0]], dtype=float)
        else:
            continue
        ok = np.isfinite(vals)
        afl, afib = vals[ok & (y == 0)], vals[ok & (y == 1)]
        if len(afl) < 10 or len(afib) < 10:
            continue
        u, p = mannwhitneyu(afl, afib, alternative="two-sided")
        rows.append({
            "feature": name, "group": r["group"], "shap": float(r["importance"]),
            "median_afl": float(np.median(afl)), "median_afib": float(np.median(afib)),
            "direction": "AFL> AFIB" if np.median(afl) > np.median(afib) else "AFIB> AFL",
            "p_value": float(p),
        })
    out = pd.DataFrame(rows)
    out.to_csv(MET / "physiology_direction_check.csv", index=False)
    print(out.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
