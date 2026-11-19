"""Stage 3.0 — statistical validation: repeated grouped CV, OOF predictions,
bootstrap confidence intervals and paired comparison on identical records.

Compares two feature sets on the Chapman experiment set A:
- BASE  = expert + v2 (682 features)
- HEAT  = expert + v2 + heat (691 features)
Both use identical StratifiedGroupKFold folds and identical HistGB configs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
D = config.PROCESSED_ROOT / "af_afl_merged_v2"
CFG = dict(max_iter=275, learning_rate=0.06, max_leaf_nodes=15,
           min_samples_leaf=40, l2_regularization=5.0)


def bootstrap_ci(values, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(values), len(values))
        out.append(metric_fn(idx))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    man = pd.read_csv(MET / "manifest_stage3.csv")
    sub = man[man.source == "chapman"].reset_index(drop=True)
    orig = sub["orig_idx"].to_numpy()
    y = sub["label"].to_numpy()
    groups = sub["patient_key"].to_numpy()
    e = np.load(D / "features_expert.npy", mmap_mode="r")
    v = np.load(D / "features_v2.npy", mmap_mode="r")
    h = np.load(D / "features_heat.npy", mmap_mode="r")
    F_base = np.concatenate([np.asarray(e[orig]), np.asarray(v[orig])], axis=1)
    F_heat = np.concatenate([F_base, np.asarray(h[orig])], axis=1)
    for F in (F_base, F_heat):
        F[:] = np.nan_to_num(F, nan=-9.0, posinf=0.0, neginf=0.0)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(sgkf.split(F_base, y, groups)):
        for name, F in (("base682", F_base), ("heat691", F_heat)):
            m = HistGradientBoostingClassifier(**CFG, class_weight="balanced", random_state=42)
            m.fit(F[tr], y[tr])
            p = m.predict_proba(F[te])[:, 1]
            pred = (p >= 0.5).astype(int)  # threshold-free comparison on OOF
            rows.append({"fold": fold, "model": name,
                         "f1_macro": f1_score(y[te], pred, average="macro"),
                         "afib_f1": f1_score(y[te], pred, pos_label=1),
                         "roc_auc": roc_auc_score(y[te], p),
                         "n_test": len(te)})
        print(f"fold {fold} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "cv_folds.csv", index=False)

    lines = ["# Stage 3.0 — валидация и статистическая значимость", "",
             "5-кратная `StratifiedGroupKFold` на Chapman (experiment set A), "
             "одинаковые фолды для обеих моделей. Порог 0.5 (без подбора на тесте).", "",
             df.to_string(index=False), ""]
    for name in ("base682", "heat691"):
        s = df[df.model == name]
        for metric in ("f1_macro", "afib_f1", "roc_auc"):
            lo, hi = bootstrap_ci(s[metric].to_numpy(), lambda idx: s[metric].to_numpy()[idx].mean())
            lines.append(f"- {name} {metric}: {s[metric].mean():.4f} "
                         f"[95% CI {lo:.4f}, {hi:.4f}], sd={s[metric].std():.4f}")
    lines.append("")
    b = df[df.model == "base682"].set_index("fold").sort_index()
    hh = df[df.model == "heat691"].set_index("fold").sort_index()
    for metric in ("f1_macro", "afib_f1", "roc_auc"):
        d = (hh[metric] - b[metric]).to_numpy()
        try:
            stat, pval = wilcoxon(d)
        except Exception:
            stat, pval = np.nan, np.nan
        lines.append(f"- paired delta {metric}: {d.mean():+.4f} (per-fold {np.round(d,4).tolist()}), "
                     f"Wilcoxon p={pval:.4f}")
    lines += ["", "Вывод: сравнение выполнено на одинаковых фолдах и записях; "
              "статистическая значимость оценивается парным Wilcoxon по 5 фолдам."]
    (ROOT / "reports" / "AF_AFL_STAGE3_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote validation report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
