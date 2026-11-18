"""Train & evaluate classic ML binary models (AF vs AFL) on the af_afl dataset.

Sub-task of the ELTA ECG arrhythmia project (companion to ``train_af_afl_binary.py``).

Pipeline
--------
1. Binary dataset: ``data/processed/af_afl/`` (AF=1 / AFL=0).
2. Robust median/MAD normalisation (fit on train) -> raw 12-lead signals
   -> hand-crafted feature vectors via ``ecgdet.features.extract_features``.
3. Classic models: RandomForest, HistGradientBoosting, LogisticRegression, KNN, SVC.
4. Normalisations: robust_median_mad, none. Augmentation: noise + scale jitter on features.
5. Evaluate: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, log loss, MCC, kappa,
   confusion matrix, per-class metrics, ROC / PR curves, loss-free comparison figures.
6. Persist metrics / figures / text report under ``reports/``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)
from ecgdet import normalize, features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures" / "af_afl"
MET = ROOT / "reports" / "metrics" / "af_afl"
MODELS_DIR = ROOT / "reports" / "models" / "af_afl"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

BINARY_CLASSES = ["AFL", "AF"]   # index 0 -> AFL, index 1 -> AF


# ----------------------------------------------------------------- data
def load_split(split: str):
    X = np.load(PROCESSED / "af_afl" / f"X_{split}.npy").astype(np.float32)
    y = np.load(PROCESSED / "af_afl" / f"y_{split}.npy").astype(np.int64)
    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")


def to_feat_matrix(Xn: np.ndarray, fs: float):
    """(N, L, S) -> (N, F) float matrix + sorted key list."""
    key_sets = []
    mats = []
    for i in range(Xn.shape[0]):
        sig2d = np.transpose(Xn[i], (1, 0))  # (S, L) -> (n_samples, n_leads)
        fd = features.extract_features(sig2d, fs=fs, lead_names=features.LEAD_ORDER_12)
        key_sets.append(sorted(fd.keys()))
    keys = sorted({k for ks in key_sets for k in ks})
    mats = []
    for i in range(Xn.shape[0]):
        sig2d = np.transpose(Xn[i], (1, 0))
        fd = features.extract_features(sig2d, fs=fs, lead_names=features.LEAD_ORDER_12)
        mats.append([float(fd.get(k, np.nan)) for k in keys])
    M = np.array(mats, dtype=np.float64)
    return M, keys


FS = 500.0


def fit_norm(X):
    cfg = normalize.RobustNormConfig(method="median_mad")
    fitted = normalize.fit_robust_norm(np.transpose(X, (0, 2, 1)), cfg)
    return fitted, normalize.apply_robust_norm


def apply_norm(Xn: np.ndarray, fitted) -> np.ndarray:
    out = normalize.apply_robust_norm(np.transpose(Xn, (0, 2, 1)), fitted)
    return np.transpose(out, (0, 2, 1))


fitted_medmad = fit_norm(X_train)[0]
Xn_train = apply_norm(X_train, fitted_medmad)
Xn_val = apply_norm(X_val, fitted_medmad)
Xn_test = apply_norm(X_test, fitted_medmad)

Xtr_med, keys_med = to_feat_matrix(Xn_train, FS)
Xva_med, _ = to_feat_matrix(Xn_val, FS)
Xte_med, _ = to_feat_matrix(Xn_test, FS)

fitted_none = normalize.fit_robust_norm(np.transpose(X_train, (0, 2, 1)), normalize.RobustNormConfig(method="none"))
Xn_train_n = apply_norm(X_train, fitted_none)
Xn_val_n = apply_norm(X_val, fitted_none)
Xn_test_n = apply_norm(X_test, fitted_none)
Xtr_n, keys_n = to_feat_matrix(Xn_train_n, FS)
Xva_n, _ = to_feat_matrix(Xn_val_n, FS)
Xte_n, _ = to_feat_matrix(Xn_test_n, FS)

for M in [Xtr_med, Xva_med, Xte_med, Xtr_n, Xva_n, Xte_n]:
    allnan = np.where(np.isnan(M).all(axis=0))[0]
    if len(allnan) > 0:
        M = np.delete(M, allnan, axis=1)
    med = np.nanmedian(M, axis=0)
    mask = np.isnan(M)
    if mask.any():
        M[mask] = np.take(med, np.where(mask)[1])
    M = np.nan_to_num(M.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    M = M.astype(np.float64)

rng = np.random.RandomState(42)
def augment_feats(X, y, factor=0.3, noise_std=0.005):
    if factor <= 0:
        return X.copy(), y.copy()
    n = int(len(X) * factor)
    idx = rng.choice(len(X), n, replace=True)
    Xa = X[idx].copy()
    std = np.std(Xa, axis=0, keepdims=True) + 1e-12
    Xa = Xa + rng.normal(0, noise_std, Xa.shape) * std
    Xa = Xa * rng.uniform(0.99, 1.01, Xa.shape)
    return np.concatenate([X, Xa], axis=0), np.concatenate([y, y[idx]], axis=0)

MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=800, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVC": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
}

results = []
detailed_results = []

norms = [
    ("robust_median_mad", Xtr_med, Xva_med, Xte_med, keys_med),
    ("none", Xtr_n, Xva_n, Xte_n, keys_n),
]

for nname, Xtr, Xva, Xte, keys in norms:
    Xtr_aug, y_train_aug_local = augment_feats(Xtr, y_train, factor=0.3)
    Xva_aug, _ = augment_feats(Xva, y_val, factor=0.0)
    Xte_aug, _ = augment_feats(Xte, y_test, factor=0.0)
    for aug, (Xtr_u, Xva_u, Xte_u) in [
        (False, (Xtr, Xva, Xte)),
        (True, (Xtr_aug, Xva_aug, Xte_aug)),
    ]:
        tag = f"{nname}_aug{int(aug)}"
        print(f"=== Normalization: {nname} | Aug: {aug} ===", flush=True)
        for mname, clf in MODELS.items():
            try:
                clf.fit(Xtr_u, y_train_aug_local if aug else y_train)
                yp = clf.predict(Xte_u)
                yprob = clf.predict_proba(Xte_u)[:, 1]
                ll = float(log_loss(y_test, np.clip(yprob, 1e-12, 1 - 1e-12)))
                auc = float(roc_auc_score(y_test, yprob))
                auprc = float(average_precision_score(y_test, yprob))
                acc = float(accuracy_score(y_test, yp))
                bal_acc = float(balanced_accuracy_score(y_test, yp))
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, yp, average="macro", zero_division=0)
                prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, yp, average="weighted", zero_division=0)
                mcc = float(matthews_corrcoef(y_test, yp))
                kappa = float(cohen_kappa_score(y_test, yp))
                cm = confusion_matrix(y_test.astype(int), yp.astype(int))
                fpr, tpr, _ = roc_curve(y_test, yprob)

                from sklearn.metrics import precision_recall_curve
                prec_r, rec_r, _ = precision_recall_curve(y_test, yprob)

                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=BINARY_CLASSES, yticklabels=BINARY_CLASSES)
                plt.title(f"Confusion Matrix - {tag} - {mname}")
                plt.ylabel("True"); plt.xlabel("Predicted")
                plt.tight_layout()
                plt.savefig(FIG / f"cm_classic_{mname}_{tag}.png", dpi=150)
                plt.close()
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {tag} - {mname}"); plt.legend()
                plt.tight_layout()
                plt.savefig(FIG / f"roc_classic_{mname}_{tag}.png", dpi=150)
                plt.close()
                plt.figure(figsize=(6, 5))
                plt.plot(rec_r, prec_r, label=f"AUPRC = {auprc:.4f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {tag} - {mname}"); plt.legend()
                plt.tight_layout()
                plt.savefig(FIG / f"pr_classic_{mname}_{tag}.png", dpi=150)
                plt.close()

                joblib_path = MODELS_DIR / f"{mname}_binary.joblib"
                import joblib
                joblib.dump(clf, joblib_path)

                res = {
                    "run": tag, "norm": nname, "aug": int(aug), "model": mname,
                    "accuracy": acc, "balanced_accuracy": bal_acc,
                    "precision_macro": float(prec), "recall_macro": float(rec), "f1_macro": float(f1),
                    "precision_weighted": float(prec_w), "recall_weighted": float(rec_w), "f1_weighted": float(f1_w),
                    "mcc": mcc, "kappa": kappa, "log_loss": ll, "auc_roc": auc, "auc_pr": auprc,
                }
                results.append(res)
                for c_idx, c in enumerate(BINARY_CLASSES):
                    detailed_results.append({
                        "run": tag, "norm": nname, "aug": int(aug), "model": mname, "class": c, "label": c_idx,
                        "precision": float(precision_score(y_test, yp, pos_label=c_idx)),
                        "recall": float(recall_score(y_test, yp, pos_label=c_idx)),
                        "f1": float(f1_score(y_test, yp, pos_label=c_idx)),
                        "support": int((y_test == c_idx).sum()),
                    })
                print(f"    {mname}: F1={f1:.4f} Acc={acc:.4f} AUC={auc:.4f} AUPRC={auprc:.4f} LL={ll:.4f}")
            except Exception as e:
                print(f"    ERROR {mname}: {e}")

res_df = pd.DataFrame(results)
res_df.to_csv(MET / "summary_classic_binary.csv", index=False)
det_df = pd.DataFrame(detailed_results)
det_df.to_csv(MET / "per_class_metrics_classic_binary.csv", index=False)

lines = [
    "AF vs AFL Binary Classic ML Evaluation Report",
    "=" * 50,
    "Task: binary recognition of AF (atrial fibrillation) vs AFL (atrial flutter) on raw 12-lead ECG.",
    f"Classes (binary): {BINARY_CLASSES}  (AF=1, AFL=0)",
    f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}",
    "",
]
for _, r in res_df.iterrows():
    lines.append(f"Run: {r['run']} | Model: {r['model']}")
    lines.append(f"  Acc: {r['accuracy']:.4f} | BalAcc: {r['balanced_accuracy']:.4f} | F1_macro: {r['f1_macro']:.4f} | F1_w: {r['f1_weighted']:.4f}")
    lines.append(f"  Prec_macro: {r['precision_macro']:.4f} | Rec_macro: {r['recall_macro']:.4f} | MCC: {r['mcc']:.4f} | Kappa: {r['kappa']:.4f} | LogLoss: {r['log_loss']:.4f} | AUC: {r['auc_roc']:.4f} | AUPRC: {r['auc_pr']:.4f}")
    lines.append("")

lines.append("Per-class metrics (test set):")
for _, row in det_df.iterrows():
    lines.append(f"  Run: {row['run']} | Model: {row['model']} | Class: {row['class']} | Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f} | F1: {row['f1']:.4f} | Support: {row['support']}")

(MET / "full_evaluation_binary.txt").write_text("\n".join(lines), encoding="utf-8")
print("Done!")
