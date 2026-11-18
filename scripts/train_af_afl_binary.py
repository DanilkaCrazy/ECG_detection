"""Train binary AF-vs-AFL arrhythmia models (CNN1D / ResNet1D) and evaluate.

Sub-task of the ELTA ECG arrhythmia project:
    recognise AF (atrial fibrillation) vs AFL (atrial flutter) on raw
    12-lead ECG signals, so the pair can later be inserted into the overall
    8-class arrhythmia model (see ``eval_full_cascade.py``).

Pipeline
--------
1. Binary dataset: ``data/processed/af_afl/`` (built by ``build_af_afl_dataset``).
2. Robust median/MAD normalisation fitted on the training split.
3. Train CNN1D / ResNet1D with binary cross-entropy (sigmoid head, binary=True).
4. Evaluate: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, log loss, MCC, kappa,
   confusion matrix, per-class metrics, ROC curve, PR curve, loss curves.
5. Persist models / metrics / figures / text report under ``reports/``.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)

from ecgdet import normalize
from ecgdet.models import CNN1D, ResNet1D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures" / "af_afl"
MET = ROOT / "reports" / "metrics" / "af_afl"
MODELS_DIR = ROOT / "reports" / "models" / "af_afl"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}", flush=True)

# ----------------------------------------------------------------- config
EPOCHS = 15
PATIENCE = 5
LR = 1e-3
BATCH_SIZE = 32
RANDOM_STATE = 42
TARGET_FS = 500.0
TARGET_LEN = 2500
N_LEADS = 12
BINARY_CLASSES = ["AFL", "AF"]   # index 0 -> AFL, index 1 -> AF
BINARY_POS = 1                    # 1 == AF


# ----------------------------------------------------------------- data
def load_split(split: str):
    X = np.load(ROOT / "data" / "processed" / "af_afl" / f"X_{split}.npy").astype(np.float32)
    y = np.load(ROOT / "data" / "processed" / "af_afl" / f"y_{split}.npy").astype(np.int64)
    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)


# ----------------------------------------------------------------- normalisation (robust median/MAD, fit on train)
norm_cfg = normalize.RobustNormConfig(method="median_mad")
norm_fitted = normalize.fit_robust_norm(np.transpose(X_train, (0, 2, 1)), norm_cfg)  # (N, S, L)


def robust_apply(X_lr):
    X3 = np.transpose(X_lr, (0, 2, 1))  # (N, S, L)
    out3 = normalize.apply_robust_norm(X3, norm_fitted)  # (N, S, L)
    return np.transpose(out3, (0, 2, 1))  # back to (N, L, S)


X_train_n = robust_apply(X_train)
X_val_n = robust_apply(X_val)
X_test_n = robust_apply(X_test)
print("norm stats applied", flush=True)


# ----------------------------------------------------------------- augmentation
def augment_signal(sig: np.ndarray) -> np.ndarray:
    sig_aug = sig.copy()
    sig_aug += np.random.normal(0, 0.05, sig_aug.shape).astype(np.float32)
    scale = np.random.uniform(0.95, 1.05)
    sig_aug *= scale
    return sig_aug.astype(np.float32)


def augment_dataset(X, y, factor=0.5):
    n = int(len(X) * factor)
    idx = np.random.choice(len(X), n, replace=True)
    X_aug = np.array([augment_signal(X[i]) for i in idx], dtype=np.float32)
    y_aug = y[idx]
    return np.concatenate([X, X_aug], axis=0), np.concatenate([y, y_aug], axis=0)


# ----------------------------------------------------------------- torch utils
def make_loaders(Xtr, ytr, Xvn, yvn, Xtn, ytn, cw_arr):
    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
    Xvn_t = torch.tensor(np.asarray(Xvn, dtype=np.float32))
    yvn_t = torch.tensor(np.asarray(yvn, dtype=np.int64))
    Xtn_t = torch.tensor(np.asarray(Xtn, dtype=np.float32))
    ytn_t = torch.tensor(np.asarray(ytn, dtype=np.int64))

    train_ds = TensorDataset(Xtr_t, ytr_t)
    val_ds = TensorDataset(Xvn_t, yvn_t)
    test_ds = TensorDataset(Xtn_t, ytn_t)

    per_sample_w = np.array([cw_arr[int(t)] for t in ytr_t.numpy()], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.tensor(per_sample_w, dtype=torch.float32), len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb.view(-1, 1).float())
                val_loss += loss.item() * xb.size(0)
                prob = torch.sigmoid(out).view(-1)
                all_probs.append(prob.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_prob = np.concatenate(all_probs)
        val_true = np.concatenate(all_targets)
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_auc = float(roc_auc_score(val_true, val_prob)) if len(np.unique(val_true)) > 1 else 0.5
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))
        val_acc = float(accuracy_score(val_true, val_pred))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        print(f"    Epoch {epoch+1}: loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}", flush=True)
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, test_loader):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            prob = torch.sigmoid(out).view(-1)
            all_probs.append(prob.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probs)


# ----------------------------------------------------------------- run loop
results = []
detailed_results = []

for aug in [False, True]:
    if aug:
        Xtr, ytr = augment_dataset(X_train_n, y_train, factor=0.5)
        tag = f"aug{int(aug)}"
    else:
        Xtr, ytr = X_train_n.copy(), y_train.copy()
        tag = f"aug{int(aug)}"

    class_counts = np.bincount(ytr.astype(np.int64), minlength=2)
    cw_arr = len(ytr) / (class_counts.astype(np.float64) * len(class_counts))
    cw_tensor = torch.tensor([float(cw_arr[1])], dtype=torch.float32).to(DEVICE)

    train_loader, val_loader, test_loader = make_loaders(Xtr, ytr, X_val_n, y_val, X_test_n, y_test, cw_arr.tolist())

    for mname, model in [
        ("CNN1D", CNN1D(num_classes=2, in_channels=N_LEADS, dropout=0.3, binary=True)),
        ("ResNet1D", ResNet1D(num_classes=2, in_channels=N_LEADS, dropout=0.3, binary=True)),
    ]:
        print(f"  Training {mname} (aug={aug})...", flush=True)
        model = model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(weight=cw_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        start = time.time()
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS, patience=PATIENCE)
        tr_time = time.time() - start
        y_true, y_prob = evaluate_model(model, test_loader)
        y_pred = (y_prob >= 0.5).astype(np.int64)

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        mcc = float(matthews_corrcoef(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        ll = float(log_loss(y_true, np.clip(y_prob, 1e-12, 1 - 1e-12)))
        auc = float(roc_auc_score(y_true, y_prob))
        auc_pr = float(average_precision_score(y_true, y_prob))

        cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec_r, rec_r, _ = precision_recall_curve(y_true, y_prob)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=BINARY_CLASSES, yticklabels=BINARY_CLASSES)
        plt.title(f"Confusion Matrix - {mname} - {tag}")
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(FIG / f"cm_{mname}_{tag}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {mname} - {tag}"); plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"roc_{mname}_{tag}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(rec_r, prec_r, label=f"AUPRC = {auc_pr:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve - {mname} - {tag}"); plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"pr_{mname}_{tag}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.title(f"{mname} - Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"curves_{mname}_{tag}.png", dpi=150)
        plt.close()

        torch.save({
            "model_state": model.state_dict(),
            "model_class": mname,
            "num_classes": 2,
            "classes": BINARY_CLASSES,
            "binary": True,
            "task": "AF_vs_AFL_binary",
            "augmentation": int(aug),
            "history": history,
            "norm_method": norm_cfg.method,
            "mean": None,
            "std": None,
        }, MODELS_DIR / f"{mname}_{tag}.pt")

        res = {
            "run": f"{mname}_{tag}", "model": mname, "task": "AF_vs_AFL_binary", "aug": int(aug),
            "accuracy": acc, "balanced_accuracy": bal_acc,
            "precision_macro": float(prec), "recall_macro": float(rec), "f1_macro": float(f1),
            "precision_weighted": float(prec_w), "recall_weighted": float(rec_w), "f1_weighted": float(f1_w),
            "mcc": mcc, "kappa": kappa, "log_loss": ll, "auc_roc": auc, "auc_pr": auc_pr,
            "train_time": tr_time,
        }
        results.append(res)

        for c_idx, c in enumerate(BINARY_CLASSES):
            detailed_results.append({
                "run": f"{mname}_{tag}", "model": mname, "task": "AF_vs_AFL_binary", "aug": int(aug),
                "class": c, "label": c_idx,
                "precision": float(precision_score(y_true, y_pred, pos_label=c_idx)),
                "recall": float(recall_score(y_true, y_pred, pos_label=c_idx)),
                "f1": float(f1_score(y_true, y_pred, pos_label=c_idx)),
                "support": int((y_true == c_idx).sum()),
            })

        print(f"  {mname} (aug={aug}): F1_macro={f1:.4f} Acc={acc:.4f} AUC={auc:.4f} AUPRC={auc_pr:.4f} LL={ll:.4f} | ({tr_time:.1f}s)", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(MET / "summary_binary.csv", index=False)
    det_df = pd.DataFrame(detailed_results)
    det_df.to_csv(MET / "per_class_metrics_binary.csv", index=False)


# ----------------------------------------------------------------- text report
lines = [
    "AF vs AFL Binary Model Evaluation Report",
    "=" * 50,
    "Task: binary recognition of AF (atrial fibrillation) vs AFL (atrial flutter) on raw 12-lead ECG.",
    "Purpose: specialised head that will later be inserted into the overall 8-class arrhythmia model.",
    f"Classes (binary): {BINARY_CLASSES}  (AF=1, AFL=0)",
    f"Train: {len(X_train_n)}, Val: {len(X_val_n)}, Test: {len(X_test_n)}",
    f"Normalization: robust median/MAD fitted on train (method={norm_cfg.method})",
    "",
]
for _, r in res_df.iterrows():
    lines.append(f"Run: {r['run']}")
    lines.append(f"  Acc: {r['accuracy']:.4f} | BalAcc: {r['balanced_accuracy']:.4f} | F1_macro: {r['f1_macro']:.4f} | F1_w: {r['f1_weighted']:.4f}")
    lines.append(f"  Prec_macro: {r['precision_macro']:.4f} | Rec_macro: {r['recall_macro']:.4f} | MCC: {r['mcc']:.4f} | Kappa: {r['kappa']:.4f} | LogLoss: {r['log_loss']:.4f} | AUC: {r['auc_roc']:.4f} | AUPRC: {r['auc_pr']:.4f} | Time: {r['train_time']:.1f}s")
    lines.append("")

lines.append("Per-class metrics (test set):")
for _, row in det_df.iterrows():
    lines.append(f"  Run: {row['run']} | Class: {row['class']} | Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f} | F1: {row['f1']:.4f} | Support: {row['support']}")

(MET / "full_evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nDone!", flush=True)
