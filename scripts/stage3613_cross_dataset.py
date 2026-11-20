"""Stage 3.6.13 — cross-dataset generalisation with UDA methods.

For each ordered source->target pair we compare:
  none   : source-only MLP
  CORAL  : second-order statistics alignment
  MMD    : multi-kernel MMD
  DANN   : gradient-reversal domain adversary

Training uses source TRAIN labels + target TRAIN *features only* (unsupervised
domain adaptation).  The classification threshold is selected on source VAL
and then frozen; target labels are used only for the final metrics.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.domain_adaptation import gradient_reversal, coral_loss, mmd_loss  # noqa: E402
from ecgdet.external_evaluation import select_threshold, evaluate_predictions  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "stage3613"
MET.mkdir(parents=True, exist_ok=True)
PAIRS = [("chapman", "ptbxl"), ("chapman", "cpsc2018_extra"),
         ("ptbxl", "chapman"), ("cpsc2018_extra", "chapman")]
METHODS = ["none", "CORAL", "MMD", "DANN"]
EPOCHS = 25


class Net(nn.Module):
    def __init__(self, d, with_domain=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2))
        self.cls = nn.Linear(64, 1)
        self.dom = nn.Linear(64, 2) if with_domain else None

    def forward(self, x, alpha=0.0):
        z = self.enc(x)
        logit = self.cls(z).squeeze(-1)
        dom = self.dom(gradient_reversal(z, alpha)) if self.dom else None
        return logit, z, dom


def prep(X, fit_idx):
    X = np.asarray(X, dtype=np.float64)
    med = np.nan_to_num(np.nanmedian(X[fit_idx], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit_idx], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit_idx], axis=0)) / sc, -20, 20).astype(np.float32)


def run_pair(meta, X, source, target, method):
    src = meta.source.to_numpy()
    spl = meta.split.to_numpy()
    y = meta.label.to_numpy().astype(np.float32)
    fit = np.where((src == source) & (spl == "train"))[0]
    val = np.where((src == source) & (spl == "val"))[0]
    tgt = np.where(src == target)[0]
    if len(fit) < 20 or len(val) < 10 or len(tgt) < 20:
        return None
    if np.unique(y[fit]).size < 2 or np.unique(y[val]).size < 2 or np.unique(y[tgt]).size < 2:
        return None

    xp = prep(X, fit)
    torch.manual_seed(42)
    net = Net(xp.shape[1], with_domain=(method == "DANN"))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((y[fit] == 0).sum() /
                                                              max((y[fit] == 1).sum(), 1))]))
    Xs, ys = torch.tensor(xp[fit]), torch.tensor(y[fit])
    Xt = torch.tensor(xp[tgt])
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(fit))
        for k in range(0, len(perm), 128):
            idx = perm[k:k + 128]
            alpha = min(1.0, 2.0 / (1 + np.exp(-10 * ep / EPOCHS)) - 1.0)
            logit, z, dom = net(Xs[idx], alpha)
            loss = bce(logit, ys[idx])
            if method in ("CORAL", "MMD"):
                t = Xt[torch.randperm(len(Xt))[:len(idx)]]
                _, zt, _ = net(t)
                if method == "CORAL":
                    loss = loss + 0.1 * coral_loss(z, zt)
                else:
                    loss = loss + 0.1 * mmd_loss(z, zt)
            elif method == "DANN":
                t = Xt[torch.randperm(len(Xt))[:len(idx)]]
                _, _, dt = net(t, alpha)
                dom_t = torch.cat([torch.zeros(len(idx)), torch.ones(len(t))]).long()
                loss = loss + nn.functional.cross_entropy(torch.cat([dom, dt]), dom_t)
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        sv = torch.sigmoid(net(torch.tensor(xp[val]))[0]).numpy()
        tv = torch.sigmoid(net(torch.tensor(xp[tgt]))[0]).numpy()
    # Probability direction guard: a sub-chance source-val AUC is a training
    # failure, not something to fix by flipping the target predictions.
    thr = select_threshold(y[val], sv)
    src_metrics = evaluate_predictions(y[val], sv, thr)
    tgt_metrics = evaluate_predictions(y[tgt], tv, thr)
    return {
        "source": source, "target": target, "method": method,
        "n_source_val": len(val), "n_target": len(tgt),
        "threshold": thr,
        "source_val_roc_auc": src_metrics["roc_auc"],
        "target_roc_auc": tgt_metrics["roc_auc"],
        "target_f1_macro": tgt_metrics["f1_macro"],
        "target_afib_f1": tgt_metrics["afib_f1"],
        "target_afl_f1": tgt_metrics["afl_f1"],
        "target_balanced_accuracy": tgt_metrics["balanced_accuracy"],
    }


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "features_advanced_expert.npy", mmap_mode="r")
    rows = []
    for src, tgt in PAIRS:
        for method in METHODS:
            r = run_pair(meta, X, src, tgt, method)
            if r:
                rows.append(r)
                print(f"{src:14}->{tgt:14} {method:5} AUC={r['target_roc_auc']:.3f} "
                      f"F1={r['target_f1_macro']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "cross_dataset_results.csv", index=False)
    (MET / "cross_dataset_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
