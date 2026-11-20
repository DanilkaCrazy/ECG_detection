"""Stage 3.6.11 — train the modular ECG network with the strict protocol.

fit = Chapman train; threshold = Chapman val; control = Chapman test;
external sources evaluated with the frozen threshold.  Attention weights are
persisted as an interpretability artifact (NOT a physiological decomposition).
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
from ecgdet.modular_ecg import ModularECGNet, modular_loss  # noqa: E402
from ecgdet.external_evaluation import evaluate_predictions, select_threshold, assert_probability_direction  # noqa: E402
from ecgdet.label_audit import check_label_direction  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
U = ROOT / "data" / "processed" / "af_afl_stage3611"
MET = ROOT / "reports" / "metrics" / "stage3611"
EXP = ROOT / "experiments" / "stage_3_6_11" / "modular"
MET.mkdir(parents=True, exist_ok=True)
EXP.mkdir(parents=True, exist_ok=True)
EPOCHS, BATCH, LR = 14, 64, 1e-3


def norm(x: np.ndarray) -> np.ndarray:
    """Stage 3.6.15 fix: per-LEAD normalisation over TIME (axis=-1)."""
    from ecgdet.signal_normalization import normalize_per_lead_time
    return normalize_per_lead_time(x)


def main() -> int:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    t0 = time.time()
    meta = pd.read_csv(D / "meta.csv")
    check_label_direction(meta)
    d = np.load(U / "inputs.npz")
    rr, mask = d["rr"], d["mask"]
    atrial, raw = norm(d["atrial"]), norm(d["raw"])
    y = meta.label.to_numpy().astype(np.float32)
    src = meta.source.to_numpy()
    spl = meta.split.to_numpy()

    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    test = np.where((src == "chapman") & (spl == "test"))[0]
    exts = {s: np.where(src == s)[0] for s in ("ptbxl", "cpsc2018_extra")}

    def tensors(idx):
        return (torch.from_numpy(raw[idx]), torch.from_numpy(atrial[idx]),
                torch.from_numpy(rr[idx]), torch.from_numpy(mask[idx]),
                torch.from_numpy(y[idx]))

    Xf, Af, Rf, Mf, Yf = tensors(fit)
    dl = DataLoader(TensorDataset(Xf, Af, Rf, Mf, Yf), batch_size=BATCH, shuffle=True)

    net = ModularECGNet(n_leads=12, n_samples=1250, n_rr=40, dim=32)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best, state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        total = 0.0
        for xb, ab, rb, mb, yb in dl:
            logits, aux = net(xb, ab, rb, mb)
            loss, _ = modular_loss(logits, yb, aux)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        net.eval()
        with torch.no_grad():
            lv = float(np.mean([torch.nn.functional.binary_cross_entropy_with_logits(
                net(torch.from_numpy(raw[i:i + 256]), torch.from_numpy(atrial[i:i + 256]),
                    torch.from_numpy(rr[i:i + 256]), torch.from_numpy(mask[i:i + 256]))[0],
                torch.from_numpy(y[i:i + 256])).item()
                for i in range(0, len(val), 256)]))
        if lv < best:
            best, state = lv, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 4 == 0 or ep == EPOCHS - 1:
            print(f"ep {ep+1} train={total/len(dl):.4f} val={lv:.4f}", flush=True)
    net.load_state_dict(state); net.eval()

    def predict(idx):
        out, wts = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                j = idx[i:i + 256]
                logits, aux = net(torch.from_numpy(raw[j]), torch.from_numpy(atrial[j]),
                                  torch.from_numpy(rr[j]), torch.from_numpy(mask[j]))
                out.append(torch.sigmoid(logits).numpy())
                wts.append(aux["module_weights"].numpy())
        return np.concatenate(out), np.concatenate(wts)

    sv, _ = predict(val)
    assert_probability_direction(y[val], sv)
    thr = select_threshold(y[val], sv)
    results = {"threshold": thr, "epochs": EPOCHS, "best_val_loss": float(best)}
    for name, idx in [("chapman_test", test), *exts.items()]:
        s, w = predict(idx)
        m = evaluate_predictions(y[idx], s, thr)
        results[name] = {k: v for k, v in m.items() if k != "confusion_matrix"}
        pd.DataFrame({
            "record_id": meta.record_id.iloc[idx].to_numpy(),
            "source": meta.source.iloc[idx].to_numpy(),
            "y_true": y[idx], "p_afib": s,
            "y_pred": (s >= thr).astype(int), "threshold": thr,
            "w_rhythm": w[:, 0], "w_atrial": w[:, 1], "w_morphology": w[:, 2],
        }).to_csv(MET / f"modular_routed_pred_{name}.csv", index=False)
        if name == "chapman_test":
            results["mean_module_weights"] = {
                "rhythm": float(w[:, 0].mean()), "atrial": float(w[:, 1].mean()),
                "morphology": float(w[:, 2].mean()),
            }

    results["total_seconds"] = time.time() - t0
    (MET / "modular_results_routed.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    torch.save({"state_dict": net.state_dict(), "threshold": thr,
                "epochs": EPOCHS, "note": "explicit atrial residual routing + per-lead norm + antialias"},
               EXP / "modular_routed.pt")
    print(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
