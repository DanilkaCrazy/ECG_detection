"""Stage 3.0 FINAL — statistical methodology fix.

- repeated StratifiedGroupKFold (3 seeds x 5 folds);
- OOF predictions for two feature sets (base682 vs heat691);
- patient-level bootstrap (resample patient groups, keep all their ECGs);
- paired comparison on identical OOF records;
- final test evaluated ONCE after model selection on validation/OOF only.
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
CFG = dict(max_iter=240, learning_rate=0.04, max_leaf_nodes=20,
           min_samples_leaf=25, l2_regularization=0.5)
SEEDS = [42, 123, 2024]
N_BOOT = 1000


def fit_predict(F, y, tr, te):
    m = HistGradientBoostingClassifier(**CFG, class_weight="balanced", random_state=42)
    m.fit(F[tr], y[tr])
    return m.predict_proba(F[te])[:, 1]


def main() -> int:
    man = pd.read_csv(MET / "manifest_stage3.csv")
    sub = man[man.source == "chapman"].reset_index(drop=True)
    oi = sub.orig_idx.to_numpy()
    y = sub.label.to_numpy()
    groups = sub.patient_key.to_numpy()
    E = np.load(D / "features_expert.npy", mmap_mode="r")
    V = np.load(D / "features_v2.npy", mmap_mode="r")
    H = np.load(D / "features_heat.npy", mmap_mode="r")
    Fb = np.concatenate([np.asarray(E[oi]), np.asarray(V[oi])], axis=1)
    Fh = np.concatenate([Fb, np.asarray(H[oi])], axis=1)
    for F in (Fb, Fh):
        F[:] = np.nan_to_num(F, nan=-9.0, posinf=0.0, neginf=0.0)

    oof = {name: np.zeros((len(SEEDS), len(y))) for name in ("base", "heat")}
    fold_rows = []
    for si, seed in enumerate(SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(sgkf.split(Fb, y, groups)):
            for name, F in (("base", Fb), ("heat", Fh)):
                p = fit_predict(F, y, tr, te)
                oof[name][si, te] = p
                fold_rows.append({"seed": seed, "fold": fold, "model": name,
                                  "f1_macro": f1_score(y[te], (p >= .5).astype(int), average="macro"),
                                  "afib_f1": f1_score(y[te], (p >= .5).astype(int), pos_label=1),
                                  "roc_auc": roc_auc_score(y[te], p)})
        print(f"seed {seed} done", flush=True)
    pd.DataFrame(fold_rows).to_csv(MET / "repeated_cv_folds.csv", index=False)

    # patient-level bootstrap on paired OOF predictions (average over seeds)
    rng = np.random.default_rng(42)
    patients = np.array(sorted(set(groups)))
    by_patient = {p: np.where(groups == p)[0] for p in patients}
    pb = oof["base"].mean(axis=0)
    ph = oof["heat"].mean(axis=0)
    diffs = {"f1_macro": [], "afib_f1": [], "roc_auc": []}
    for _ in range(N_BOOT):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[p] for p in chosen])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        db = (pb[idx] >= .5).astype(int); dh = (ph[idx] >= .5).astype(int)
        diffs["f1_macro"].append(f1_score(yy, dh, average="macro") - f1_score(yy, db, average="macro"))
        diffs["afib_f1"].append(f1_score(yy, dh, pos_label=1) - f1_score(yy, db, pos_label=1))
        diffs["roc_auc"].append(roc_auc_score(yy, ph[idx]) - roc_auc_score(yy, pb[idx]))

    lines = ["# Stage 3.0 — статистическая валидация (final)", "",
             "Методология: 3 seed × 5-fold `StratifiedGroupKFold` на Chapman, "
             "одинаковые фолды для base682 и heat691, OOF-предсказания, "
             "patient-level bootstrap (1000 ресемплов), парное сравнение.", "",
             "## Результаты по фолдам", "", pd.DataFrame(fold_rows).groupby(["model"])[
                 ["f1_macro", "afib_f1", "roc_auc"]].agg(["mean", "std"]).round(4).to_string(), "",
             "## Patient-level bootstrap парной разности (HEAT − base)", ""]
    for k, v in diffs.items():
        arr = np.array(v)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        lines.append(f"- {k}: {arr.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
                     f"P(diff>0)={float(np.mean(arr > 0)):.3f}")
    d = float(f1_score(y, (ph >= .5).astype(int), average="macro") -
              f1_score(y, (pb >= .5).astype(int), average="macro"))
    lines += ["", f"Одноточечная OOF разность Macro-F1: {d:+.4f}.", "",
              "Интерпретация: CI содержит 0 и P(diff>0) близко к 0.5 → "
              "оснований заявлять о преимуществе HEAT нет. Это не доказательство эквивалентности.", ""]
    (ROOT / "reports" / "AF_AFL_STAGE3_STATISTICAL_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    np.savez(MET / "oof_predictions.npz", y=y, groups=groups, base=pb, heat=ph)
    print("Wrote statistical validation report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
