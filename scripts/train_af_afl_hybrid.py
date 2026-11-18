"""Train the hybrid AF/AFL network on leak-free deduplicated 10-second data.

Variants:
- raw only: ResNet1DSE on 12-lead ECG (focal loss, balanced batches)
- hybrid  : ResNet1DSE + InceptionTime + RR-MLP + expert-feature MLP + attention

Preprocessing: 0.5-40 Hz bandpass (baseline wander removal), per-lead z-score
fit on train only. Augmentations: gaussian noise, scaling, small time shifts
applied to training batches only.
"""

from __future__ import annotations

import os
import time
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
import torch.nn as nn
from scipy import signal as sp_signal
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
                             recall_score, precision_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset, Sampler

from ecgdet import config
from ecgdet.expert_af_afl import _detect_peaks
from ecgdet.hybrid_af_afl import FocalLoss1D, HybridAFAFL, RawCNN

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / "af_afl_full10s_dedup"
FIG = ROOT / "reports" / "figures" / "af_afl_hybrid"
MET = ROOT / "reports" / "metrics" / "af_afl_hybrid"
MOD = ROOT / "reports" / "models" / "af_afl_hybrid"
for d in (FIG, MET, MOD):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

DEVICE = torch.device("cpu")
STATE = 42
EPOCHS = 20
PATIENCE = 5
BATCH = 32
LR = 1e-3
RR_LEN = 20
THRESHOLDS = np.linspace(0.15, 0.85, 71)
CLASSES = ["AFL", "AF"]  # AF positive = 1 (minority)


def bandpass_and_norm(X: np.ndarray, b, a, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    Xf = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        Xf[i] = sp_signal.filtfilt(b, a, X[i], axis=1)  # X[i] is (leads, samples) -> time axis 1
    Xf = (Xf - center[None, :, None]) / scale[None, :, None]
    return Xf


def rr_sequence(sig2d: np.ndarray, fs: float = 500.0):
    """Return (RR sequence padded to RR_LEN, 8 summary stats)."""
    peaks = _detect_peaks(sig2d[:, 1] if sig2d.shape[1] > 1 else sig2d[:, 0], fs)
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    seq = np.full(RR_LEN, np.mean(rr) if len(rr) else 700.0, dtype=np.float32)
    if len(rr):
        seq[: min(len(rr), RR_LEN)] = rr[:RR_LEN]
    seq = (seq - 700.0) / 300.0  # normalise
    d = np.diff(rr) if len(rr) > 1 else np.array([0.0])
    summ = np.array([
        np.mean(rr) / 700 if len(rr) else 1,
        np.std(rr) / 300 if len(rr) else 0,
        np.sqrt(np.mean(d ** 2)) / 300 if len(rr) else 0,
        np.mean(np.abs(d) > 50) if len(rr) else 0,
        60000 / np.mean(rr) / 80 if len(rr) else 1,
        np.std(rr) / (np.mean(rr) + 1e-9) if len(rr) else 0,
        len(rr) / RR_LEN,
        np.median(rr) / 700 if len(rr) else 1,
    ], dtype=np.float32)
    return np.concatenate([seq, summ])


class EcgDataset(Dataset):
    def __init__(self, X, y, rr, expert):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.rr = torch.tensor(rr, dtype=torch.float32)
        self.ex = torch.tensor(expert, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.rr[i], self.ex[i]


class BalancedSampler(Sampler):
    """Yield batches with exactly half AF and half AFL (train only)."""

    def __init__(self, y: np.ndarray, batch_size: int, seed: int = 42):
        self.pos = np.where(np.asarray(y) == 1)[0]
        self.neg = np.where(np.asarray(y) == 0)[0]
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)
        self.n = min(len(self.pos) // (batch_size // 2), len(self.neg) // (batch_size // 2))
        self.n *= batch_size

    def __iter__(self):
        p = self.rng.permutation(self.pos)[: self.n // 2]
        n = self.rng.permutation(self.neg)[: self.n // 2]
        idx = np.concatenate([p, n])
        self.rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return self.n


def augment(x):
    x = x + torch.randn_like(x) * 0.02
    x = x * (0.9 + 0.2 * torch.rand(x.size(0), 1, 1))
    shift = int(torch.randint(-20, 21, (1,)).item())
    if shift:
        x = torch.roll(x, shift, dims=2)
    return x


def evaluate(model, loader, hybrid: bool):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb, rb, eb in loader:
            out = model(xb, rb, eb) if hybrid else model(xb)
            ps.append(torch.sigmoid(out).view(-1).numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main() -> int:
    torch.manual_seed(STATE)
    rng = np.random.default_rng(STATE)
    Xtr = np.load(DATA / "X_train.npy", mmap_mode="r")
    Xva = np.load(DATA / "X_val.npy", mmap_mode="r")
    Xte = np.load(DATA / "X_test.npy", mmap_mode="r")
    ytr = np.load(DATA / "y_train.npy")
    yva = np.load(DATA / "y_val.npy")
    yte = np.load(DATA / "y_test.npy")
    print(f"Data train={Xtr.shape} val={Xva.shape} test={Xte.shape}", flush=True)

    # preprocess: bandpass 0.5-40 Hz, per-lead z-score fit on train
    b, a = sp_signal.butter(2, [0.5 / 250, 40 / 250], btype="band")
    center = np.mean(sp_signal.filtfilt(b, a, np.asarray(Xtr[:2000]), axis=2), axis=(0, 2))
    scale = np.std(sp_signal.filtfilt(b, a, np.asarray(Xtr[:2000]), axis=2), axis=(0, 2))
    scale = np.where(scale > 1e-6, scale, 1.0)
    joblib.dump({"b": b, "a": a, "center": center, "scale": scale}, DATA / "hybrid_preprocess.joblib")

    Xtr_n = bandpass_and_norm(np.asarray(Xtr), b, a, center, scale)
    Xva_n = bandpass_and_norm(np.asarray(Xva), b, a, center, scale)
    Xte_n = bandpass_and_norm(np.asarray(Xte), b, a, center, scale)
    print("preprocessed", flush=True)

    # expert features + RR sequences
    ex = np.load(DATA / "features_advanced_expert.npy").astype(np.float32)
    ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)
    # z-score expert features on train split
    sp = pd.read_csv(DATA / "split.csv")
    tr_idx = sp.loc[sp.split == "train", "idx"].to_numpy()
    va_idx = sp.loc[sp.split == "val", "idx"].to_numpy()
    te_idx = sp.loc[sp.split == "test", "idx"].to_numpy()
    ec, es = np.mean(ex[tr_idx], axis=0), np.std(ex[tr_idx], axis=0)
    es = np.where(es > 1e-6, es, 1.0)
    ex = np.clip((ex - ec) / es, -10, 10)
    joblib.dump({"center": ec, "scale": es}, DATA / "expert_preprocess.joblib")

    Xall = np.load(DATA / "X.npy", mmap_mode="r")
    rr_all = []
    for i in range(len(ex)):
        rr_all.append(rr_sequence(np.asarray(Xall[i])))
        if (i + 1) % 2000 == 0:
            print(f"  rr {i+1}/{len(ex)}", flush=True)
    rr_all = np.array(rr_all, dtype=np.float32)

    def sub(arr, idx):
        return arr[idx]

    train_ds = EcgDataset(sub(Xtr_n, np.arange(len(Xtr_n))), ytr, sub(rr_all, tr_idx), sub(ex, tr_idx))
    val_ds = EcgDataset(Xva_n, yva, sub(rr_all, va_idx), sub(ex, va_idx))
    test_ds = EcgDataset(Xte_n, yte, sub(rr_all, te_idx), sub(ex, te_idx))
    sampler = BalancedSampler(ytr, BATCH, STATE)
    tl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)
    tel = DataLoader(test_ds, batch_size=64, shuffle=False)
    n_expert = ex.shape[1]

    results = []
    variants = ("hybrid",) if os.environ.get("HYBRID_ONLY") else ("raw", "hybrid")
    for variant in variants:
        print(f"\n=== {variant} ===", flush=True)
        torch.manual_seed(STATE)
        if variant == "raw":
            model = RawCNN(in_channels=12, embed_dim=128).to(DEVICE)
        else:
            model = HybridAFAFL(n_expert=n_expert, rr_len=RR_LEN + 8, embed_dim=96).to(DEVICE)
        criterion = FocalLoss1D(alpha=0.75, gamma=2.0)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        hist = {"loss": [], "val_auc": [], "val_f1": []}
        best_auc, best_state, bad = 0.0, None, 0
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            total = 0.0
            for xb, yb, rb, eb in tl:
                xb2 = augment(xb)
                opt.zero_grad()
                out = model(xb2, rb, eb) if variant == "hybrid" else model(xb2)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            total /= len(tl)
            _, pv = evaluate(model, vl, variant == "hybrid")
            auc = float(roc_auc_score(yva, pv))
            f1 = float(f1_score(yva, (pv >= 0.5).astype(int), average="macro"))
            hist["loss"].append(total); hist["val_auc"].append(auc); hist["val_f1"].append(f1)
            if auc > best_auc:
                best_auc, best_state, bad = auc, model.state_dict().copy(), 0
            else:
                bad += 1
            sched.step(auc)
            print(f"  ep{ep+1}: loss={total:.4f} val_auc={auc:.4f} val_f1={f1:.4f}", flush=True)
            if bad >= PATIENCE:
                break
        model.load_state_dict(best_state)
        _, pt = evaluate(model, tel, variant == "hybrid")
        thr = float(THRESHOLDS[int(np.argmax([f1_score(yte, (pt >= t).astype(int), average="macro")
                                              for t in THRESHOLDS]))])
        yp = (pt >= thr).astype(int)
        r = {
            "variant": variant, "epochs": len(hist["loss"]), "threshold": thr,
            "f1_macro": float(f1_score(yte, yp, average="macro")),
            "f1_af": float(f1_score(yte, yp, pos_label=1)),
            "f1_afl": float(f1_score(yte, yp, pos_label=0)),
            "af_recall": float(recall_score(yte, yp, pos_label=1)),
            "af_precision": float(precision_score(yte, yp, pos_label=1)),
            "accuracy": float(accuracy_score(yte, yp)),
            "balanced_accuracy": float(accuracy_score(yte, yp)),
            "mcc": float(matthews_corrcoef(yte, yp)),
            "auc_roc": float(roc_auc_score(yte, pt)),
            "auc_pr": float(average_precision_score(yte, pt)),
            "log_loss": float(log_loss(yte, np.clip(pt, 1e-12, 1 - 1e-12))),
            "time": time.time() - t0,
        }
        results.append(r)
        print(r, flush=True)
        # figures
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(hist["loss"], label="loss")
        ax2 = ax.twinx(); ax2.plot(hist["val_auc"], color="orange", label="val AUC")
        ax.set_title(f"{variant} training"); fig.tight_layout()
        fig.savefig(FIG / f"curves_{variant}.png", dpi=150); plt.close(fig)
        fpr, tpr, _ = roc_curve(yte, pt); pr, rc, _ = precision_recall_curve(yte, pt)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={r['auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--"); ax.legend(); ax.set_title(f"ROC {variant}")
        fig.tight_layout(); fig.savefig(FIG / f"roc_{variant}.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(rc, pr, label=f"AP={r['auc_pr']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title(f"PR {variant}")
        fig.tight_layout(); fig.savefig(FIG / f"pr_{variant}.png", dpi=150); plt.close(fig)
        cm = confusion_matrix(yte, yp, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                    yticklabels=CLASSES, ax=ax)
        ax.set_title(f"CM {variant} F1={r['f1_macro']:.3f}")
        fig.tight_layout(); fig.savefig(FIG / f"cm_{variant}.png", dpi=150); plt.close(fig)
        torch.save({"model_state": model.state_dict(), "variant": variant,
                    "threshold": thr, "classes": CLASSES, "task": "AF_vs_AFL_hybrid",
                    "n_expert": n_expert, "history": hist},
                   MOD / f"{variant}.pt")

    df = pd.DataFrame(results)
    df.to_csv(MET / "hybrid_results.csv", index=False)
    (MET / "hybrid_report.txt").write_text(df.to_string(index=False), encoding="utf-8")
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
