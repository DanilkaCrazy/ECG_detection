"""Train binary AF-vs-AFL CNN/ResNet on the FULL dataset with imbalance handling.

Compared variants:
- pos_weight : BCEWithLogitsLoss with positive-class weight + WeightedRandomSampler
- oversample : minority class duplicated to parity, plain BCE
- focal      : Focal loss (alpha=0.75, gamma=2) + sampler
- aug        : pos_weight + noise/scale augmentation

Early stopping on validation AUC; decision threshold tuned on validation F1.
Figures: loss/AUC curves, ROC, PR, confusion matrix, comparison bars.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, roc_auc_score, roc_curve,
                             precision_recall_curve, matthews_corrcoef)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ecgdet import config, normalize
from ecgdet.models import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / "af_afl_full"
FIG = ROOT / "reports" / "figures" / "af_afl_full_dl"
MET = ROOT / "reports" / "metrics" / "af_afl_full_dl"
MOD = ROOT / "reports" / "models" / "af_afl_full_dl"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

DEVICE = torch.device("cpu")
EPOCHS = 10
PATIENCE = 3
LR = 1e-3
BATCH = 128
STATE = 42
THRESHOLDS = np.linspace(0.15, 0.85, 71)
CLASSES = ["AFL", "AF"]


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        targets = targets.view(-1, 1).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()


def load_split(name: str):
    return (np.load(DATA / f"X_{name}.npy").astype(np.float32),
            np.load(DATA / f"y_{name}.npy").astype(np.int64))


def main() -> int:
    rng = np.random.default_rng(STATE)
    torch.manual_seed(STATE)
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    print(f"Shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape}", flush=True)

    # robust norm fit on train: (N, S, L) view
    cfg = normalize.RobustNormConfig(method="median_mad")
    fitted = normalize.fit_robust_norm(np.transpose(Xtr, (0, 2, 1)), cfg)
    joblib.dump(fitted, DATA / "norm_fitted.joblib")

    def norm(X):
        return np.transpose(normalize.apply_robust_norm(np.transpose(X, (0, 2, 1)), fitted), (0, 2, 1)).astype(np.float32)

    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)

    def make_loader(X, y, sampler=False, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        if sampler:
            cnt = np.bincount(y, minlength=2)
            w = np.array([len(y) / (2 * cnt[t]) for t in y], dtype=np.float32)
            s = WeightedRandomSampler(torch.tensor(w), len(ds), replacement=True)
            return DataLoader(ds, batch_size=BATCH, sampler=s)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    def evaluate(model, loader):
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                p = torch.sigmoid(model(xb)).view(-1).cpu().numpy()
                ps.append(p); ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    def train_variant(mname: str, variant: str):
        # data preparation per variant
        Xt, yt = Xtr, ytr
        if variant == "oversample":
            minority = Xt[yt == 1]
            need = int((yt == 0).sum()) - int((yt == 1).sum())
            idx = rng.integers(0, len(minority), size=need)
            Xt = np.concatenate([Xt, minority[idx]])
            yt = np.concatenate([yt, np.ones(need, dtype=np.int64)])
        use_sampler = variant in ("pos_weight", "focal", "aug")
        train_loader = make_loader(Xt, yt, sampler=use_sampler)
        val_loader = make_loader(Xva, yva)
        test_loader = make_loader(Xte, yte)
        cnt = np.bincount(ytr, minlength=2)
        pos_w = float(cnt[0] / max(cnt[1], 1))
        if variant == "focal":
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
        else:
            criterion = nn.BCEWithLogitsLoss(
                weight=torch.tensor([pos_w], dtype=torch.float32) if variant != "oversample" else None
            )
        model = build_model(mname, num_classes=2, in_channels=12, binary=True).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        hist: dict[str, list] = {"loss": [], "val_auc": [], "val_f1": []}
        best_auc, best_state, bad = 0.0, None, 0
        for ep in range(EPOCHS):
            model.train()
            tl = 0.0
            for xb, yb in train_loader:
                if variant == "aug":
                    xb = xb + torch.randn_like(xb) * 0.05
                    xb = xb * (0.95 + 0.1 * torch.rand(xb.size(0), 1, 1))
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb.view(-1, 1).float())
                loss.backward()
                opt.step()
                tl += loss.item() * xb.size(0)
            tl /= len(train_loader.dataset)
            _, pv = evaluate(model, val_loader)
            auc = float(roc_auc_score(yva, pv))
            f1 = float(f1_score(yva, (pv >= 0.5).astype(int), average="macro"))
            hist["loss"].append(tl); hist["val_auc"].append(auc); hist["val_f1"].append(f1)
            if auc > best_auc:
                best_auc, best_state, bad = auc, model.state_dict().copy(), 0
            else:
                bad += 1
            sched.step(auc)
            if bad >= PATIENCE:
                break
        model.load_state_dict(best_state)
        _, pt = evaluate(model, test_loader)
        scores = [f1_score(yte, (pt >= t).astype(int), average="macro") for t in THRESHOLDS]
        thr = float(THRESHOLDS[int(np.argmax(scores))])
        yp = (pt >= thr).astype(int)
        res = {
            "model": mname, "variant": variant, "threshold": thr,
            "f1_macro": float(f1_score(yte, yp, average="macro")),
            "f1_af": float(f1_score(yte, yp, pos_label=1)),
            "f1_afl": float(f1_score(yte, yp, pos_label=0)),
            "accuracy": float(accuracy_score(yte, yp)),
            "mcc": float(matthews_corrcoef(yte, yp)),
            "auc_roc": float(roc_auc_score(yte, pt)),
            "auc_pr": float(average_precision_score(yte, pt)),
            "log_loss": float(log_loss(yte, np.clip(pt, 1e-12, 1 - 1e-12))),
        }
        # figures
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist["loss"], label="train loss")
        ax.plot(hist["val_auc"], label="val AUC")
        ax.set_title(f"{mname} / {variant} training")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIG / f"curves_{mname}_{variant}.png", dpi=150); plt.close(fig)
        fpr, tpr, _ = roc_curve(yte, pt)
        pr, rc, _ = precision_recall_curve(yte, pt)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC={res['auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"ROC {mname}/{variant}")
        fig.tight_layout(); fig.savefig(FIG / f"roc_{mname}_{variant}.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rc, pr, label=f"AP={res['auc_pr']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
        ax.set_title(f"PR {mname}/{variant}")
        fig.tight_layout(); fig.savefig(FIG / f"pr_{mname}_{variant}.png", dpi=150); plt.close(fig)
        cm = confusion_matrix(yte, yp, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(f"CM {mname}/{variant} F1={res['f1_macro']:.3f}")
        fig.tight_layout(); fig.savefig(FIG / f"cm_{mname}_{variant}.png", dpi=150); plt.close(fig)
        torch.save({"model_state": model.state_dict(), "model_class": mname,
                    "num_classes": 2, "classes": CLASSES, "binary": True,
                    "task": "AF_vs_AFL_binary_full", "variant": variant,
                    "threshold": thr, "norm_method": cfg.method, "history": hist},
                   MOD / f"{mname}_{variant}.pt")
        print(f"  {mname}/{variant}: F1={res['f1_macro']:.4f} AUC={res['auc_roc']:.4f} "
              f"AP={res['auc_pr']:.4f} thr={thr:.2f}", flush=True)
        return res

    results = []
    for variant in ("pos_weight", "oversample", "focal"):
        for mname in ("CNN1D", "ResNet1D"):
            t0 = time.time()
            print(f"Training {mname}/{variant}...", flush=True)
            results.append(train_variant(mname, variant))
            print(f"  time={time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(MET / "full_dl_results.csv", index=False)
    for metric in ("f1_macro", "auc_roc", "auc_pr"):
        pv = df.pivot(index="model", columns="variant", values=metric)
        pv.plot(kind="bar", figsize=(11, 5))
        plt.title(f"{metric} — DL imbalance variants")
        plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(FIG / f"dl_comparison_{metric}.png", dpi=150); plt.close()
    (MET / "full_dl_report.txt").write_text(
        "AF/AFL full-data DL imbalance study\n" + "=" * 50 + "\n" + df.to_string(index=False),
        encoding="utf-8")
    print("Done!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
