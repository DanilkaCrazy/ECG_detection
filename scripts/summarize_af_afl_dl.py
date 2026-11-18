"""Rebuild the DL results table from saved checkpoints (no retraining)."""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, roc_auc_score)

from ecgdet import config, normalize
from ecgdet.models import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = config.PROCESSED_ROOT / "af_afl_full"
MOD = ROOT / "reports" / "models" / "af_afl_full_dl"
MET = ROOT / "reports" / "metrics" / "af_afl_full_dl"
FIG = ROOT / "reports" / "figures" / "af_afl_full_dl"
MET.mkdir(parents=True, exist_ok=True)


def main() -> int:
    Xte = np.load(DATA / "X_test.npy").astype(np.float32)
    yte = np.load(DATA / "y_test.npy").astype(int)
    fitted = joblib.load(DATA / "norm_fitted.joblib")
    xn = np.transpose(normalize.apply_robust_norm(np.transpose(Xte, (0, 2, 1)), fitted), (0, 2, 1)).astype(np.float32)
    xt = torch.tensor(xn, dtype=torch.float32)

    rows = []
    for ckpt in sorted(MOD.glob("*.pt")):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(ck["model_class"], num_classes=2, in_channels=12, binary=True)
        model.load_state_dict(ck["model_state"])
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(xt), 256):
                ps.append(torch.sigmoid(model(xt[i:i + 256])).view(-1).numpy())
        p = np.concatenate(ps)
        thr = float(ck.get("threshold", 0.5))
        yp = (p >= thr).astype(int)
        rows.append({
            "model": ck["model_class"],
            "variant": ck.get("variant", "?"),
            "threshold": thr,
            "f1_macro": float(f1_score(yte, yp, average="macro")),
            "f1_af": float(f1_score(yte, yp, pos_label=1)),
            "f1_afl": float(f1_score(yte, yp, pos_label=0)),
            "accuracy": float(accuracy_score(yte, yp)),
            "mcc": float(matthews_corrcoef(yte, yp)),
            "auc_roc": float(roc_auc_score(yte, p)),
            "auc_pr": float(average_precision_score(yte, p)),
            "log_loss": float(log_loss(yte, np.clip(p, 1e-12, 1 - 1e-12))),
        })
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv(MET / "full_dl_results.csv", index=False)
    (MET / "full_dl_report.txt").write_text(
        "AF/AFL full-data DL imbalance study\n" + "=" * 50 + "\n" + df.to_string(index=False),
        encoding="utf-8")
    for metric in ("f1_macro", "auc_roc", "auc_pr"):
        pv = df.pivot(index="model", columns="variant", values=metric)
        pv.plot(kind="bar", figsize=(11, 5))
        plt.title(f"{metric} — DL imbalance variants")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG / f"dl_comparison_{metric}.png", dpi=150)
        plt.close()
    print("Done!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
