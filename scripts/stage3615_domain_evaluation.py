"""Stage 3.6.15 — corrected UDA protocol with a SEALED target test set.

Three regimes are reported separately:
  source_only        : no target data at all
  transductive_uda   : adaptation on unlabeled target + evaluation on SAME target
  inductive_uda      : adaptation on unlabeled target subset, evaluation on a
                       disjoint locked target subset
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
from ecgdet.domain_evaluation import split_target_domain, validate_target_isolation  # noqa: E402
from ecgdet.external_evaluation import select_threshold, evaluate_predictions  # noqa: E402

SRC = ROOT / "data" / "processed" / "af_afl_stage34"
D = ROOT / "data" / "processed" / "af_afl_stage3614"
MET = ROOT / "reports" / "metrics" / "stage3615"
PAIRS = [("chapman", "ptbxl"), ("chapman", "cpsc2018_extra"),
         ("ptbxl", "chapman"), ("cpsc2018_extra", "chapman")]
METHODS = ["none", "CORAL", "MMD", "DANN"]
EPOCHS = 20


class Net(nn.Module):
    def __init__(self, d, dom=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2))
        self.cls = nn.Linear(64, 1)
        self.dom = nn.Linear(64, 2) if dom else None

    def forward(self, x, alpha=0.0):
        z = self.enc(x)
        return self.cls(z).squeeze(-1), z, (self.dom(gradient_reversal(z, alpha))
                                            if self.dom else None)


def prep(X, fit):
    X = np.asarray(X, float)
    med = np.nan_to_num(np.nanmedian(X[fit], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit], axis=0)) / sc, -20, 20).astype(np.float32)


def fit_and_score(Xp, y, fit, val, adapt, eval_idx, method, seed=42):
    torch.manual_seed(seed)
    net = Net(Xp.shape[1], dom=(method == "DANN"))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pos = max(float((y[fit] == 0).sum() / max((y[fit] == 1).sum(), 1)), 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    Xs, ys = torch.tensor(Xp[fit]), torch.tensor(y[fit].astype(np.float32))
    Xa = torch.tensor(Xp[adapt]) if len(adapt) else None
    for ep in range(EPOCHS):
        net.train()
        for k in range(0, len(fit), 128):
            idx = np.arange(k, min(k + 128, len(fit)))
            alpha = min(1.0, 2.0 / (1 + np.exp(-10 * ep / EPOCHS)) - 1.0)
            logit, z, dom = net(Xs[idx], alpha)
            loss = bce(logit, ys[idx])
            if Xa is not None and len(Xa) > 1:
                t = Xa[torch.randperm(len(Xa))[:len(idx)]]
                _, zt, _ = net(t, alpha)
                if method == "CORAL":
                    loss = loss + 0.1 * coral_loss(z, zt)
                elif method == "MMD":
                    loss = loss + 0.1 * mmd_loss(z, zt)
                elif method == "DANN":
                    dt = net(t, alpha)[2]
                    dom_t = torch.cat([torch.zeros(len(idx)), torch.ones(len(t))]).long()
                    loss = loss + nn.functional.cross_entropy(torch.cat([dom, dt]), dom_t)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        sv = torch.sigmoid(net(torch.tensor(Xp[val]))[0]).numpy()
        ev = torch.sigmoid(net(torch.tensor(Xp[eval_idx]))[0]).numpy()
    thr = select_threshold(y[val], sv)
    m = evaluate_predictions(y[eval_idx], ev, thr)
    return {"threshold": thr, "roc_auc": m["roc_auc"], "f1_macro": m["f1_macro"],
            "afib_f1": m["afib_f1"], "afl_f1": m["afl_f1"]}


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    split = pd.read_csv(D / "split.csv")
    prov = pd.read_csv(D / "diagnosis_provenance.csv")
    rows_all = prov.row_in_stage34.to_numpy(dtype=int)
    F = np.load(SRC / "features_advanced_expert.npy", mmap_mode="r")[rows_all]
    y = meta.label.to_numpy()
    src = meta.source.to_numpy()
    spl = split.split.to_numpy()

    results = []
    for source, target in PAIRS:
        fit = np.where((src == source) & (spl == "train"))[0]
        val = np.where((src == source) & (spl == "val"))[0]
        tgt_meta = meta[src == target].reset_index(drop=True)
        tgt_global = np.where(src == target)[0]
        if len(fit) < 20 or len(val) < 10 or len(tgt_global) < 20:
            continue
        if np.unique(y[fit]).size < 2 or np.unique(y[val]).size < 2 or np.unique(y[tgt_global]).size < 2:
            continue
        adapt_local, test_local = split_target_domain(tgt_meta, adaptation_fraction=0.5, seed=42)
        assert not validate_target_isolation(adapt_local, test_local, tgt_meta)
        adapt = tgt_global[adapt_local]
        locked = tgt_global[test_local]
        Xp = prep(F, fit)
        for method in METHODS:
            # Inductive: adapt only on the adaptation half, evaluate on locked test.
            r_ind = fit_and_score(Xp, y, fit, val, adapt, locked, method)
            results.append({"source": source, "target": target, "method": method,
                            "regime": "inductive_uda", "n_adapt": len(adapt),
                            "n_eval": len(locked), **r_ind})
            # Source-only: identical inputs but no target adaptation.
            if method == "none":
                results.append({"source": source, "target": target, "method": method,
                                "regime": "source_only", "n_adapt": 0,
                                "n_eval": len(locked), **r_ind})
            # Transductive: old behaviour, adapt+evaluate on all target rows.
            if method in METHODS[1:]:
                r_tr = fit_and_score(Xp, y, fit, val, tgt_global, tgt_global, method)
                results.append({"source": source, "target": target, "method": method,
                                "regime": "transductive_uda", "n_adapt": len(tgt_global),
                                "n_eval": len(tgt_global), **r_tr})

    df = pd.DataFrame(results)
    df.to_csv(MET / "domain_evaluation_results.csv", index=False)
    (MET / "domain_evaluation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
