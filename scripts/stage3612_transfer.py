"""Stage 3.6.12 — linear probing and fine-tuning of the SSL encoder.

Compares three protocols on identical splits:
  ssl_linear_probe : frozen encoder + logistic head
  ssl_finetune     : encoder + head trained jointly
  supervised_scratch : same architecture trained from scratch with labels

Threshold always on Chapman val; external sources never tune it.
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.foundation_adapter import MaskedSignalEncoder  # noqa: E402
from ecgdet.external_evaluation import evaluate_predictions, select_threshold  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
U = ROOT / "data" / "processed" / "af_afl_stage3611"
M = ROOT / "reports" / "models" / "stage3612"
MET = ROOT / "reports" / "metrics" / "stage3612"
MET.mkdir(parents=True, exist_ok=True)
EPOCHS, BATCH = 8, 64


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def embeddings(net, x):
    out = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(x), 256):
            out.append(net(torch.from_numpy(x[i:i + 256]))[0].numpy())
    return np.concatenate(out)


def prep(X, fit_idx):
    med = np.nan_to_num(np.nanmedian(X[fit_idx], axis=0))
    X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X[fit_idx], [25, 75], axis=0)
    sc = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return np.clip((X - np.median(X[fit_idx], axis=0)) / sc, -20, 20)


def main() -> int:
    torch.manual_seed(42)
    meta = pd.read_csv(D / "meta.csv")
    raw = norm(np.load(U / "inputs.npz")["raw"])
    y = meta.label.to_numpy().astype(np.float32)
    src, spl = meta.source.to_numpy(), meta.split.to_numpy()
    fit = np.where((src == "chapman") & (spl == "train"))[0]
    val = np.where((src == "chapman") & (spl == "val"))[0]
    test = np.where((src == "chapman") & (spl == "test"))[0]
    exts = {s: np.where(src == s)[0] for s in ("ptbxl", "cpsc2018_extra")}

    state = torch.load(M / "ssl_encoder.pt", map_location="cpu", weights_only=False)
    ssl = MaskedSignalEncoder(12, 1250, 64)
    ssl.load_state_dict(state["state_dict"])
    results = {"ecg_fm_backend": state.get("ecg_fm_preflight", {})}

    # --- linear probe ---
    Ef = embeddings(ssl, raw[fit]); Ev = embeddings(ssl, raw[val])
    Et = embeddings(ssl, raw[test])
    Ee = {k: embeddings(ssl, raw[v]) for k, v in exts.items()}
    ef, ev = prep(Ef, np.arange(len(Ef))), prep(Ev, np.arange(len(Ev)))
    et = prep(Et, np.arange(len(Et)))
    ee = {k: prep(v, np.arange(len(v))) for k, v in Ee.items()}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ef, y[fit])
    sv = lr.predict_proba(ev)[:, 1]
    thr = select_threshold(y[val], sv)
    linear_test_score = lr.predict_proba(et)[:, 1]
    block = {"chapman_test": evaluate_predictions(y[test], linear_test_score, thr)}
    for k, v in ee.items():
        block[k] = evaluate_predictions(y[exts[k]], lr.predict_proba(v)[:, 1], thr)
    results["ssl_linear_probe"] = {"threshold": thr, **{
        k: {q: m[q] for q in ("f1_macro", "afib_f1", "roc_auc", "pr_auc", "brier")}
        for k, m in block.items()}}

    # --- fine-tune / supervised scratch ---
    test_scores = {}
    for mode in ("ssl_finetune", "supervised_scratch"):
        net = MaskedSignalEncoder(12, 1250, 64)
        if mode == "ssl_finetune":
            net.load_state_dict(state["state_dict"])
        head = nn.Linear(64, 1)
        opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
        dl = DataLoader(TensorDataset(torch.from_numpy(raw[fit]), torch.from_numpy(y[fit])),
                        batch_size=BATCH, shuffle=True)
        for _ in range(EPOCHS):
            net.train()
            for xb, yb in dl:
                z = net(xb)[0]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    head(z).squeeze(-1), yb,
                    pos_weight=torch.tensor([float((y[fit] == 0).sum() / max((y[fit] == 1).sum(), 1))]))
                opt.zero_grad(); loss.backward(); opt.step()

        def score(idx):
            net.eval()
            z = embeddings(net, raw[idx])
            with torch.no_grad():
                return torch.sigmoid(head(torch.from_numpy(z)).squeeze(-1)).numpy()

        s_val = score(val)
        t = select_threshold(y[val], s_val)
        out = {"threshold": t}
        for name, idx in [("chapman_test", test), *exts.items()]:
            m = evaluate_predictions(y[idx], score(idx), t)
            out[name] = {q: m[q] for q in ("f1_macro", "afib_f1", "roc_auc", "pr_auc", "brier")}
        results[mode] = out
        test_scores[mode] = score(test)

    # Paired cluster bootstrap: is SSL pretraining actually better than scratch?
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(42)
    keys = meta.patient_key.to_numpy()[test]
    by_key = {k: np.where(keys == k)[0] for k in np.unique(keys)}
    diffs = []
    for _ in range(1000):
        sidx = np.concatenate([by_key[k] for k in rng.choice(list(by_key), len(by_key), True)])
        base = f1_score(y[test][sidx], (test_scores["supervised_scratch"][sidx] >=
                                        results["supervised_scratch"]["threshold"]).astype(int),
                        average="macro")
        new = f1_score(y[test][sidx], (test_scores["ssl_finetune"][sidx] >=
                                       results["ssl_finetune"]["threshold"]).astype(int),
                       average="macro")
        diffs.append(new - base)
    arr = np.asarray(diffs)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    results["bootstrap_ssl_finetune_minus_scratch"] = {
        "mean": float(arr.mean()), "lo": float(lo), "hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }
    diffs_lp = []
    for _ in range(1000):
        sidx = np.concatenate([by_key[k] for k in rng.choice(list(by_key), len(by_key), True)])
        base = f1_score(y[test][sidx], (test_scores["supervised_scratch"][sidx] >=
                                        results["supervised_scratch"]["threshold"]).astype(int),
                        average="macro")
        new = f1_score(y[test][sidx], (linear_test_score[sidx] >= thr).astype(int),
                       average="macro")
        diffs_lp.append(new - base)
    al = np.asarray(diffs_lp)
    ll, hl = np.percentile(al, [2.5, 97.5])
    results["bootstrap_linear_probe_minus_scratch"] = {
        "mean": float(al.mean()), "lo": float(ll), "hi": float(hl),
        "significant": bool(ll > 0 or hl < 0),
    }

    (MET / "transfer_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
