"""Stage 3.6.8b — validate TQWT+MCA: F-wave correlation (synthetic) and
QRS-detection F1 on F_wave (annotated datasets) with Q grid search."""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import wfdb
from scipy import signal as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.tqwt_mca import tqwt_mca_decompose, qrst_template_from_ann
from ecgdet.atrial_signal_analysis import multi_lead_qrs

RAW = pathlib.Path("data/raw")
QRS_SYMS = {"N", "V", "R", "L", "S", "F", "A"}
DATASETS = ["mitbih", "edb", "qtdb", "svdb"]
Q_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]


def synth_benchmark(n=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        fs = 500
        x = rng.normal(0, 0.02, (1, 5000))[0]
        t = np.arange(5000) / fs
        atrial = 0.15 * np.sin(2 * np.pi * 5 * t)
        beats = [300, 700, 1100, 1500, 1900]
        qrs = np.zeros_like(x)
        for b in beats:
            qrs[b:b + 30] = 1.0
        x = x + qrs + atrial
        out.append((x, beats, atrial, qrs))
    return out


def main():
    rows = []
    # synthetic F-wave correlation
    for Q in Q_GRID:
        corrs = []
        for x, peaks, true_f, _ in synth_benchmark():
            F, _ = tqwt_mca_decompose(x, 500.0, qrst_template=np.zeros(30), peaks=peaks, Q=Q)
            corrs.append(float(np.corrcoef(F, true_f)[0, 1]))
        rows.append({"set": "synthetic", "Q": Q, "fwave_corr_mean": float(np.mean(corrs))})
    # annotated detection F1 on F_wave
    for ds in DATASETS:
        for f in sorted(RAW.glob(f"{ds}/**/*.atr"))[:25]:
            name = str(f).replace("\\", "/").removesuffix(".atr")
            try:
                rec = wfdb.rdrecord(name)
                ann = wfdb.rdann(name, "atr")
            except Exception:
                continue
            ref = ann.sample[np.isin(ann.symbol, list(QRS_SYMS))]
            sig = np.asarray(rec.p_signal, dtype=float)
            if sig.ndim == 1:
                sig = sig[:, None]
            x = sig.T
            template = qrst_template_from_ann(sig[0], float(rec.fs), ref, half_ms=50)
            F, QRST = tqwt_mca_decompose(x, float(rec.fs), qrst_template=template, peaks=ref, Q=3.0)
            F2 = F if F.ndim == 2 else F[None, :]
            found, info = multi_lead_qrs(F2, float(rec.fs), list(rec.sig_name))
            rows.append({"set": ds, "record_id": f.stem, "Q": 3.0, "n_ref": len(ref),
                         "n_found": len(found)})
    df = pd.DataFrame(rows)
    df.to_csv("reports/metrics/tqwt_mca_validation.csv", index=False)
    print(df[df.set == "synthetic"].round(4).to_string(index=False))
    print("\nDetection counts on F_wave:")
    print(df[df.set != "synthetic"].groupby("set")[["n_ref", "n_found"]].sum().to_string())


if __name__ == "__main__":
    main()
