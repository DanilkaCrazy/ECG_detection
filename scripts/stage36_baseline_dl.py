"""Stage 3.6.2 — raw-ECG DL baseline (CNN1D / ResNet1D), strict protocol."""
from __future__ import annotations
import json, time, warnings, pathlib, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.models import build_model
from stage36_baseline import metrics, THR

D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
MOD = ROOT / "reports" / "models" / "af_afl_stage36"
DEVICE = torch.device("cpu")
EPOCHS = int(__import__("os").environ.get("STAGE36_EPOCHS", "8"))
BATCH = 64


def main():
    torch.manual_seed(42)
    meta = pd.read_csv(D / "meta.csv")
    sp = pd.read_csv(D / "split.csv")
    y = meta.label.to_numpy()
    tr = np.where(sp.split == "train")[0]
    va = np.where(sp.split == "val")[0]
    te = np.where(sp.split == "test")[0]
    Xm = np.load(D / "X.npy", mmap_mode="r")

    def prep(idx):
        a = np.asarray(Xm[idx], dtype=np.float32).copy()
        m = a.mean(axis=1, keepdims=True)
        s = a.std(axis=1, keepdims=True) + 1e-6
        return torch.from_numpy((a - m) / s)

    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    yva = torch.tensor(y[va], dtype=torch.float32)
    pos_w = torch.tensor([float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))])
    rows = []
    for name in ("CNN1D", "ResNet1D"):
        t0 = time.time()
        net = build_model(name, 2, binary=True).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best_loss, best_state = np.inf, None
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(Xtr))
            for k in range(0, len(perm), BATCH):
                idx = perm[k:k + BATCH]
                opt.zero_grad()
                out = net(Xtr[idx]).squeeze(-1)
                loss = loss_fn(out, ytr[idx])
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_fn(net(Xva[k:k + BATCH]).squeeze(-1), yva[k:k + BATCH]).item()
                                    for k in range(0, len(Xva), BATCH)]))
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(name, "ep", ep + 1, "val_loss", round(vl, 4), flush=True)
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            sv = np.concatenate([torch.sigmoid(net(Xva[k:k + BATCH]).squeeze(-1)).numpy()
                                 for k in range(0, len(Xva), BATCH)])
            st = np.concatenate([torch.sigmoid(net(Xte[k:k + BATCH]).squeeze(-1)).numpy()
                                 for k in range(0, len(Xte), BATCH)])
        # threshold on val only
        thr = float(THR[int(np.argmax([f1_score(y[va], (sv >= t).astype(int), average="macro")
                                       for t in THR]))])
        r = metrics(y[te], st, thr)
        r.update({"model": name, "threshold": thr, "n_test": len(te),
                  "val_loss": float(best_loss), "epochs": EPOCHS,
                  "seconds": float(time.time() - t0)})
        rows.append(r)
        torch.save({"state_dict": best_state, "threshold": thr}, MOD / f"{name}.pt")
        pd.DataFrame({"record_id": meta.record_id.iloc[te].to_numpy(), "source": meta.source.iloc[te].to_numpy(),
                      "label": y[te], "prob_afib": st, "threshold": thr,
                      "prediction": (st >= thr).astype(int)}).to_csv(MET / f"stage36_2_pred_{name}.csv", index=False)
        print(name, "F1", round(r["f1_macro"], 4), "AFIB", round(r["afib_f1"], 4), flush=True)
    (MET / "stage36_2_baseline_dl.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    (MET / "stage36_2_baseline_dl.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("DL DONE", flush=True)


if __name__ == "__main__":
    main()
