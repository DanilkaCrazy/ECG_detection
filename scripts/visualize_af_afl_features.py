"""Visualise AF vs AFL feature distributions (RR irregularity, flutter activity,
spectral bands, morphology) and rank features by discrimination power."""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pointbiserialr

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / "af_afl_full"
FIG = ROOT / "reports" / "figures" / "af_afl_features"
MET = ROOT / "reports" / "metrics" / "af_afl_features"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

KEY_FEATURES = [
    "rr_rr_rmssd", "rr_rr_cv", "rr_rr_pnn50", "rr_rr_std", "rr_rr_entropy", "rr_rr_ac1",
    "rr_n_rpeaks", "rr_hr_mean", "rr_rpeak_width_mean_ms", "rr_rpeak_height_cv",
    "atrial_ii_flutter_ac_peak", "atrial_ii_flutter_rate_bpm", "atrial_ii_flutter_peak_ratio",
    "atrial_v1_flutter_ac_peak", "atrial_v1_flutter_peak_ratio",
    "spec0_psd_theta_rel", "spec1_psd_theta_rel", "spec6_psd_theta_rel", "spec1_psd_entropy",
    "atrial_dom_freq", "atrial_dom_power", "cross_corr_std",
]


def main() -> int:
    X = np.load(DATA / "features_advanced.npy")
    y = np.load(DATA / "y.npy")
    names = pd.read_csv(DATA / "feature_names.csv")["name"].tolist()
    pos = {n: i for i, n in enumerate(names)}
    df = pd.DataFrame({n: X[:, pos[n]] for n in KEY_FEATURES if n in pos})
    df["class"] = np.where(y == 1, "AF", "AFL")
    df = df.replace([np.inf, -np.inf], np.nan)

    # violin plots for the most interpretable features
    chosen = [n for n in KEY_FEATURES if n in df.columns][:12]
    ncol = 4
    nrow = (len(chosen) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.6 * nrow))
    for ax, name in zip(np.array(axes).ravel(), chosen):
        d = df[[name, "class"]].dropna()
        d = d[d[name].between(d[name].quantile(0.01), d[name].quantile(0.99))]
        sns.violinplot(data=d, x="class", y=name, ax=ax, palette=["#3498db", "#e74c3c"], inner="quartile")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("")
    plt.suptitle("AF vs AFL — distribution of key rhythm/morphology features", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "feature_violins.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # rank ALL features by Mann-Whitney effect size
    rows = []
    for j, name in enumerate(names):
        v = X[:, j]
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        a = v[y[np.isfinite(X[:, j])] == 1]
        b = v[y[np.isfinite(X[:, j])] == 0]
        if len(a) < 10 or len(b) < 10:
            continue
        if np.std(np.concatenate([a, b])) < 1e-12:
            continue
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            n1, n2 = len(a), len(b)
            r_rb = 2 * u / (n1 * n2) - 1  # rank-biserial effect size
            rows.append({"feature": name, "effect": float(r_rb), "p": float(p)})
        except Exception:
            pass
    imp = pd.DataFrame(rows).sort_values("effect", key=abs, ascending=False)
    imp.to_csv(MET / "feature_discrimination.csv", index=False)

    top = imp.head(30)
    plt.figure(figsize=(10, 9))
    sns.barplot(x="effect", y="feature", data=top,
                palette=["#e74c3c" if v > 0 else "#3498db" for v in top["effect"]])
    plt.title("Top-30 discriminative features (Mann–Whitney rank-biserial effect)")
    plt.tight_layout()
    plt.savefig(FIG / "feature_effect_ranking.png", dpi=150)
    plt.close()

    (MET / "feature_analysis_report.txt").write_text(
        "AF vs AFL feature analysis\n" + "=" * 50 + "\n"
        + "Positive effect = higher in AF; negative = higher in AFL.\n\n"
        + imp.head(60).to_string(index=False), encoding="utf-8")
    print(imp.head(15).to_string(index=False), flush=True)
    print("Done!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
