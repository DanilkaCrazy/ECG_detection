"""Stage 3.6.12 — local self-supervised masked-signal pretraining (TRAIN split only).

This is explicitly a FALLBACK, not ECG-FM transfer.  No labels and no
validation/test signals are used here.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.foundation_adapter import (  # noqa: E402
    MaskedSignalEncoder, mask_signal, ssl_reconstruction_loss, ecg_fm_preflight,
)

D = ROOT / "data" / "processed" / "af_afl_stage34"
U = ROOT / "data" / "processed" / "af_afl_stage3611"
OUT = ROOT / "reports" / "models" / "stage3612"
OUT.mkdir(parents=True, exist_ok=True)
EPOCHS, BATCH, MASK_PROB = 8, 64, 0.4


def main() -> int:
    torch.manual_seed(42)
    t0 = time.time()
    preflight = ecg_fm_preflight()
    meta = pd.read_csv(D / "meta.csv")
    raw = np.load(U / "inputs.npz")["raw"]
    fit = np.where((meta.source == "chapman") & (meta.split == "train"))[0]
    x = raw[fit].astype(np.float32)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    dl = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=BATCH, shuffle=True)

    net = MaskedSignalEncoder(n_leads=12, n_samples=1250, dim=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    hist = []
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for (xb,) in dl:
            xm, keep = mask_signal(xb, MASK_PROB)
            _, recon = net(xm)
            loss = ssl_reconstruction_loss(recon, xb, keep)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        hist.append(tot / len(dl))
        print(f"ssl ep {ep+1} loss={hist[-1]:.4f}", flush=True)

    torch.save({"state_dict": net.state_dict(), "epochs": EPOCHS,
                "mask_prob": MASK_PROB, "trained_on": "chapman/train only",
                "ecg_fm_preflight": preflight}, OUT / "ssl_encoder.pt")
    meta_out = {
        "backend": "local_ssl_fallback", "ecg_fm_preflight": preflight,
        "n_pretrain_records": len(fit), "epochs": EPOCHS, "loss_history": hist,
        "seconds": time.time() - t0,
        "warning": "This is NOT ECG-FM. fairseq backend and checkpoint are unavailable.",
    }
    (OUT / "ssl_pretrain.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta_out.items() if k != "loss_history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
