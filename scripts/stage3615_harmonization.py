"""Stage 3.6.15 — compare preprocessing variants H0-H5 on the same network.

All variants use identical model, split and threshold protocol; only the input
transformation changes.  H0 keeps physical units, H1-H4 are normalisation
variants, H5 uses a train-fitted scale applied identically to every split.
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.modular_ecg import ModularECGNet, modular_loss  # noqa: E402
from ecgdet.external_evaluation import evaluate_predictions, select_threshold  # noqa: E402
from ecgdet.signal_normalization import (  # noqa: E402
    normalize_per_lead_time, normalize_interlead, normalize_robust,
    fit_train_stats, apply_train_stats,
)

D = ROOT / "data" / "processed" / "af_afl_stage34"
U = ROOT / "data" / "processed" / "af_afl_stage3611"
MET = ROOT / "reports" / "metrics" / "stage3615"
EPOCHS, BATCH = 10, 64

VARIANTS = ["H0_none", "H1_per_lead_time", "H2_robust", "H3_interlead", "H4_train_fitted"]


def transform(x, a, fit, variant):
    if variant == "H0_none":
        return x, a
    if variant == "H1_per_lead_time":
        return normalize_per_lead_time(x), normalize_per_lead_time(a)
    if variant == "H2_robust":
        return normalize_robust(x), normalize_robust(a)
    if variant == "H3_interlead":
        return normalize_interlead(x), normalize_interlead(a)
    stats = fit_train_stats(x[fit])
    astats = fit_train_stats(a[fit])
    return apply_train_stats(x, stats), apply_train_stats(a, astats)


def main() -> int:
    torch.manual_seed(42)
    meta = pd.read_csv(D / "meta.csv")
    d = np.load(U / "inputs.npz")
    rr, mask = d["rr"], d["mask"]
    raw = d["raw"].astype(np.float32)
    atrial = d["atrial"].astype(np.float32)
    y = meta.label.to_numpy().astype(np.float32)
    src, spl = meta.source.to_numpy(), meta.split.to_numpy()
    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    test = np.where((src == "chapman") & (spl == "test"))[0]

    rows = []
    for variant in VARIANTS:
        X, A = transform(raw, atrial, fit, variant)
        net = ModularECGNet(12, 1250, 40, dim=32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        dl = DataLoader(TensorDataset(torch.from_numpy(X[fit]), torch.from_numpy(A[fit]),
                                      torch.from_numpy(rr[fit]), torch.from_numpy(mask[fit]),
                                      torch.from_numpy(y[fit])),
                        batch_size=BATCH, shuffle=True)
        for _ in range(EPOCHS):
            net.train()
            for xb, ab, rb, mb, yb in dl:
                logits, aux = net(xb, ab, rb, mb)
                loss, _ = modular_loss(logits, yb, aux)
                opt.zero_grad(); loss.backward(); opt.step()

        def score(idx):
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(idx), 256):
                    j = idx[i:i + 256]
                    out.append(torch.sigmoid(net(torch.from_numpy(X[j]),
                                                 torch.from_numpy(A[j]),
                                                 torch.from_numpy(rr[j]),
                                                 torch.from_numpy(mask[j]))[0]).numpy())
            return np.concatenate(out)
        sv = score(val)
        thr = select_threshold(y[val], sv)
        m = evaluate_predictions(y[test], score(test), thr)
        rows.append({"variant": variant, "threshold": thr,
                     "f1_macro": m["f1_macro"], "afib_f1": m["afib_f1"],
                     "roc_auc": m["roc_auc"], "brier": m["brier"]})
        print(variant, "F1", round(m["f1_macro"], 4), "AUC", round(m["roc_auc"], 4), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(MET / "harmonization_variants.csv", index=False)
    (MET / "harmonization_variants.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
