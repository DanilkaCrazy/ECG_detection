"""Stage 3.6.10 — SHAP / permutation feature importance for the best ablation model."""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stage3610_ablation import EXPERIMENTS, models, prep  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
U = ROOT / "data" / "processed" / "af_afl_stage3610"
MET = ROOT / "reports" / "metrics" / "stage3610"


def main() -> int:
    best = json.loads((MET / "ablation_summary.json").read_text(encoding="utf-8"))["best"]
    exp, mname = best["experiment"], best["model"]
    meta = pd.read_csv(D / "meta.csv")
    expert = np.load(D / "features_advanced_expert.npy", mmap_mode="r")
    Xg = np.load(U / "X.npy")
    feat = pd.read_csv(U / "feature_names.csv")
    grp = feat.group.to_numpy()

    cols = np.where(np.isin(grp, EXPERIMENTS[exp]))[0]
    X = np.column_stack([np.asarray(expert), Xg[:, cols]])
    names = [f"expert_{i}" for i in range(expert.shape[1])] + feat.name.to_numpy()[cols].tolist()
    groups = ["X"] * expert.shape[1] + grp[cols].tolist()

    fit = np.where((meta.source == "chapman") & (meta.split == "train"))[0]
    test = np.where((meta.source == "chapman") & (meta.split == "test"))[0]
    Xp, _ = prep(X, fit)
    keep = [j for j in range(Xp.shape[1]) if np.ptp(Xp[fit, j]) > 1e-9]
    Xp = Xp[:, keep]
    names = [names[j] for j in keep]
    groups = [groups[j] for j in keep]

    model = models()[mname]
    model.fit(Xp[fit], meta.label.to_numpy()[fit])
    y = meta.label.to_numpy()[test]

    rng = np.random.default_rng(42)
    sample = rng.choice(test, size=min(300, len(test)), replace=False)
    if mname == "HistGB":
        imp = permutation_importance(model, Xp[sample], y[sample], n_repeats=5,
                                     random_state=42, scoring="f1_macro").importances_mean
        method = "permutation"
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xp[sample])
        if isinstance(sv, list):
            arr = np.asarray(sv[1] if len(sv) == 2 else sv[0])
        else:
            arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr[:, :, 0]
        imp = np.abs(arr).mean(axis=0)
        method = "shap"

    fi = (pd.DataFrame({"feature": names, "group": groups, "importance": imp})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    fi.to_csv(MET / "feature_importance.csv", index=False)
    print(f"method={method} best={exp}_{mname} n_features={len(fi)}")
    print(fi.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
