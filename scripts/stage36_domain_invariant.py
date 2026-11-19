"""Stage 3.6.6 — DANN/CORAL on expert features with LOSO evaluation."""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.domain_adaptation import gradient_reversal, coral_loss
from sklearn.metrics import f1_score, roc_auc_score

D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
EPOCHS = 15
BATCH = 128


class Net(nn.Module):
    def __init__(self, n_features, n_domains):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2))
        self.cls = nn.Linear(64, 1)
        self.dom = nn.Linear(64, n_domains)

    def forward(self, x, alpha=1.0):
        z = self.enc(x)
        y = self.cls(z).squeeze(-1)
        d = self.dom(gradient_reversal(z, alpha))
        return y, d, z


def prep(X, tr):
    keep = ~np.isnan(X[tr]).all(axis=0)
    X = np.asarray(X)[:, keep]
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[tr], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return ((X - np.median(X[tr], axis=0)) / sc).astype(np.float32)


def main():
    torch.manual_seed(42)
    meta = pd.read_csv(D / "meta.csv")
    y = meta.label.to_numpy().astype(np.float32)
    src = meta.source.to_numpy()
    sources = sorted(set(src))
    X = prep(np.load(D / "features_advanced_expert.npy", mmap_mode="r"), np.arange(len(meta)))
    rows = []
    for held in sources:
        tr = np.where(src != held)[0]
        te = np.where(src == held)[0]
        Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
        Xte = torch.tensor(X[te])
        train_sources = sorted(set(src[tr]))
        dom_ids = np.array([train_sources.index(s) for s in src[tr]])
        dom_t = torch.as_tensor(dom_ids, dtype=torch.long)
        n_dom = len(train_sources)
        net = Net(X.shape[1], n_dom)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))]))
        ce = nn.CrossEntropyLoss()
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(tr))
            for k in range(0, len(perm), BATCH):
                idx = perm[k:k + BATCH]
                if len(idx) < 2 or len(torch.unique(dom_t[idx])) < 2:
                    continue  # need both domains in a batch for CORAL/DANN
                alpha = min(1.0, 2.0 / (1.0 + np.exp(-10 * ep / EPOCHS)) - 1.0)
                yh, dh, z = net(Xtr[idx], alpha)
                loss = bce(yh, ytr[idx]) + ce(dh, dom_t[idx])
                # CORAL between the two training sources
                s0 = z[dom_t[idx] == 0]
                s1 = z[dom_t[idx] == 1]
                if len(s0) > 2 and len(s1) > 2:
                    loss = loss + 0.1 * coral_loss(s0, s1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xte)[0]).numpy()
        q = (p >= 0.5).astype(int)
        rows.append({"held_out_source": str(held), "n_test": len(te),
                     "f1_macro": float(f1_score(y[te], q, average="macro")),
                     "afib_f1": float(f1_score(y[te], q, pos_label=1)),
                     "afl_f1": float(f1_score(y[te], q, pos_label=0)),
                     "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te])) > 1 else np.nan})
        print(held, "done", flush=True)
    (MET / "stage36_6_dann_loso.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(MET / "stage36_6_dann_loso.csv", index=False)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
