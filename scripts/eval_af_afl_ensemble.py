"""Ensemble the best AF/AFL heads (classic 5s, classic 10s, CNN 5s).

Recomputes validation/test probabilities for each saved model, then evaluates
simple averaging and AUC-weighted averaging with a validation-tuned threshold.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
                             roc_auc_score, roc_curve)
from scipy.stats import rankdata

from ecgdet import config, normalize
from ecgdet.features_advanced import extract_advanced_features
from ecgdet.features_advanced_v2 import extract_v2_features
from ecgdet.models import build_model

ROOT = Path(__file__).resolve().parents[1]
FIVE = config.PROCESSED_ROOT / "af_afl_full"
TEN = config.PROCESSED_ROOT / "af_afl_full10s"
FIG = ROOT / "reports" / "figures" / "af_afl_ensemble"
MET = ROOT / "reports" / "metrics" / "af_afl_ensemble"
for d in (FIG, MET):
    d.mkdir(parents=True, exist_ok=True)
THRESHOLDS = np.linspace(0.15, 0.85, 71)


def classic_probs(bundle_path: Path, data: Path, fn, names_file: str, split: str) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    prep = joblib.load(bundle_path.parent / "preprocessing.joblib")
    names = pd.read_csv(data / names_file)["name"].tolist()
    X = np.load(data / f"X_{split}.npy")
    rows = []
    for sig in X:
        d = fn(np.asarray(sig, dtype=np.float64).T, fs=500.0)
        rows.append([d.get(k, np.nan) for k in names])
    M = np.array(rows, dtype=np.float32)
    an = set(prep.get("allnan_cols", np.array([], dtype=int)).tolist())
    keep = [j for j in range(len(names)) if j not in an]
    M = M[:, keep]
    M = np.where(np.isnan(M), prep["median_fill"], M)
    M = np.clip((M - prep["center"]) / prep["scale"], -20, 20)
    return bundle["model"].predict_proba(M)[:, 1]


def dl_probs(ckpt: Path, data: Path, norm_path: Path, split: str) -> np.ndarray:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["model_class"], num_classes=2, in_channels=12, binary=True)
    model.load_state_dict(ck["model_state"])
    model.eval()
    fitted = joblib.load(norm_path)
    X = np.load(data / f"X_{split}.npy").astype(np.float32)
    xn = np.transpose(normalize.apply_robust_norm(np.transpose(X, (0, 2, 1)), fitted), (0, 2, 1)).astype(np.float32)
    xt = torch.tensor(xn, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), 256):
            out.append(torch.sigmoid(model(xt[i:i + 256])).view(-1).numpy())
    return np.concatenate(out)


def main() -> int:
    sources = [
        ("classic_5s", ROOT / "reports" / "models" / "af_afl_imbalance" / "best_model.joblib",
         FIVE, extract_advanced_features, "feature_names.csv"),
        ("classic_10s", ROOT / "reports" / "models" / "af_afl_imbalance_af_afl_full10s_v2" / "best_model.joblib",
         TEN, extract_v2_features, "feature_names_v2.csv"),
        ("cnn_5s", ROOT / "reports" / "models" / "af_afl_full_dl" / "CNN1D_pos_weight.pt",
         FIVE, None, None),
    ]
    yva = np.load(FIVE / "y_val.npy").astype(int)
    yte = np.load(FIVE / "y_test.npy").astype(int)
    pv, pt, names = {}, {}, []
    for name, path, data, fn, nf in sources:
        if not path.exists():
            print("skip missing:", path, flush=True)
            continue
        if fn is None:
            pva = dl_probs(path, data, data / "norm_fitted.joblib", "val")
            pte = dl_probs(path, data, data / "norm_fitted.joblib", "test")
        else:
            pva = classic_probs(path, data, fn, nf, "val")
            pte = classic_probs(path, data, fn, nf, "test")
        pv[name], pt[name] = pva, pte
        print(f"{name}: val AUC={roc_auc_score(yva, pva):.4f} test AUC={roc_auc_score(yte, pte):.4f}", flush=True)
        names.append(name)

    combos: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        combos[name] = (pv[name], pt[name])
    if len(names) >= 2:
        combos["mean_all"] = (np.mean([pv[n] for n in names], axis=0),
                              np.mean([pt[n] for n in names], axis=0))
        w = np.array([roc_auc_score(yva, pv[n]) - 0.5 for n in names])
        w = w / w.sum()
        combos["auc_weighted"] = (sum(w[i] * pv[n] for i, n in enumerate(names)),
                                  sum(w[i] * pt[n] for i, n in enumerate(names)))

    results = []
    for name, (a, b) in combos.items():
        thr = float(THRESHOLDS[int(np.argmax([f1_score(yva, (a >= t).astype(int), average="macro")
                                              for t in THRESHOLDS]))])
        yp = (b >= thr).astype(int)
        r = {"ensemble": name, "threshold": thr,
             "f1_macro": float(f1_score(yte, yp, average="macro")),
             "f1_af": float(f1_score(yte, yp, pos_label=1)),
             "accuracy": float(accuracy_score(yte, yp)),
             "auc_roc": float(roc_auc_score(yte, b)),
             "auc_pr": float(average_precision_score(yte, b)),
             "mcc": float(matthews_corrcoef(yte, yp)),
             "log_loss": float(log_loss(yte, np.clip(b, 1e-12, 1 - 1e-12)))}
        results.append(r)
        print(name, r, flush=True)
        fpr, tpr, _ = roc_curve(yte, b)
        pr, rc, _ = precision_recall_curve(yte, b)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{name} AUC={r['auc_roc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.title("ROC — ensembles")
        plt.tight_layout(); plt.savefig(FIG / f"roc_{name}.png", dpi=150); plt.close()
        cm = confusion_matrix(yte, yp, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns_cm = pd.DataFrame(cm, index=["AFL", "AF"], columns=["AFL", "AF"])
        sns.heatmap(sns_cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"CM {name}"); plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout(); plt.savefig(FIG / f"cm_{name}.png", dpi=150); plt.close()

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    df.to_csv(MET / "ensemble_results.csv", index=False)
    (MET / "ensemble_report.txt").write_text(df.to_string(index=False), encoding="utf-8")
    print("\nBEST ENSEMBLE:\n", df.head(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
