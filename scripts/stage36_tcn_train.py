"""Stage 3.6.7 — clinical training of TwoStreamTCN on real 12-lead ECG."""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.two_stream_tcn import TwoStreamTCN, multitask_loss
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

D = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "af_afl_stage34"
MET = pathlib.Path(__file__).resolve().parents[1] / "reports" / "metrics" / "af_afl_stage36"


def main():
    torch.manual_seed(42)
    meta = pd.read_csv(D / "meta.csv"); sp = pd.read_csv(D / "split.csv")
    y = meta.label.to_numpy().astype(np.float32)
    X = np.load(D / "X.npy", mmap_mode="r")
    tr = np.where(sp.split == "train")[0]; va = np.where(sp.split == "val")[0]; te = np.where(sp.split == "test")[0]

    def t(idx):
        a = np.asarray(X[idx], dtype=np.float32)
        a = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-6)
        return torch.from_numpy(a)

    net = TwoStreamTCN(12, 5000)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ytr = torch.tensor(y[tr]); yva = torch.tensor(y[va]); yte = torch.tensor(y[te])
    Xtr, Xva, Xte = t(tr), t(va), t(te)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((y[tr]==0).sum()/max((y[tr]==1).sum(),1))]))
    best = np.inf; best_state = None
    for ep in range(30):
        net.train()
        tot = 0
        for xb, yb in dl:
            q, a, logit = net(xb)
            loss, _ = multitask_loss(xb, q, a, logit, yb, atrial_true=None)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([bce(net(vb[0])[2], vb[1]).item() for vb in
                                DataLoader(TensorDataset(Xva, yva), batch_size=256)]))
        if vl < best:
            best = vl; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 29:
            print(f"ep {ep+1} val_loss {vl:.4f}", flush=True)
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(net(vb[0])[2]).numpy() for vb in
                            DataLoader(TensorDataset(Xte), batch_size=256)])
    q = (p >= 0.5).astype(int)
    r = dict(f1_macro=float(f1_score(yte, q, average="macro")),
             afib_f1=float(f1_score(yte, q, pos_label=1)),
             afl_f1=float(f1_score(yte, q, pos_label=0)),
             roc_auc=float(roc_auc_score(yte, p)),
             pr_auc=float(average_precision_score(yte, p)), n_test=len(te))
    (MET / "stage36_7_tcn_train.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    pd.DataFrame({"record_id": meta.record_id.iloc[te].to_numpy(), "label": yte,
                  "prob": p, "prediction": q}).to_csv(MET / "stage36_7_tcn_train.csv", index=False)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
