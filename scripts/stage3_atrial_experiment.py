"""Stage 3.0 — atrial activity experiment (V1/II/III/aVF, QRS suppression)."""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

from ecgdet import config
from ecgdet.atrial_activity import atrial_activity_features

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_merged_v2"
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
FIG = ROOT / "reports" / "figures" / "af_afl_stage3"
MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(MET / "manifest_stage3.csv")
sub = man[man.source == "chapman"].reset_index(drop=True)


def one(i):
    idx = int(sub.loc[i, "orig_idx"])
    X = np.load(D / "X.npy", mmap_mode="r")
    return atrial_activity_features(np.asarray(X[idx]).T, fs=500.0)


def main() -> int:
    rng = np.random.default_rng(42)
    pick = rng.choice(len(sub), size=min(1200, len(sub)), replace=False)
    res = Parallel(n_jobs=8, verbose=0)(delayed(one)(int(i)) for i in pick)
    keys = sorted(res[0].keys())
    M = pd.DataFrame(res)
    M["label"] = sub.loc[pick, "label"].to_numpy()
    rows = []
    for k in keys:
        a = M.loc[M.label == 1, k].dropna()
        b = M.loc[M.label == 0, k].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        eff = 2 * u / (len(a) * len(b)) - 1
        rows.append({"feature": k, "effect": float(eff), "p": float(p),
                     "mean_afib": float(a.mean()), "mean_afl": float(b.mean())})
    df = pd.DataFrame(rows).sort_values("effect", key=abs, ascending=False)
    df.to_csv(MET / "atrial_activity_effects.csv", index=False)
    top = df.head(20).sort_values("effect")
    plt.figure(figsize=(9, 7))
    plt.barh(top.feature, top.effect, color=["#e74c3c" if v > 0 else "#3498db" for v in top.effect])
    plt.title("Atrial activity features (QRS-suppressed), Mann–Whitney effect")
    plt.tight_layout(); plt.savefig(FIG / "atrial_effects.png", dpi=150); plt.close()
    lines = ["# Stage 3.0 — анализ предсердной активности", "",
             "Модуль `atrial_activity.py`: подавление QRS интерполяцией (±60 мс), "
             "полосовая фильтрация 3–15 Гц, доминирующая частота, спектральная "
             "пиковость, автокорреляционный предсердный цикл и вариабельность "
             "цикла между II/III/aVF/V1.", "",
             f"Проанализировано {len(pick)} записей Chapman.", "", df.head(25).to_string(index=False)]
    (ROOT / "reports" / "AF_AFL_STAGE3_ATRIAL.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote atrial report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
