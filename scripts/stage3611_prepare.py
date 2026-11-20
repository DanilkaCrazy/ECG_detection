"""Stage 3.6.11 — prepare modular-network inputs (RR, atrial residual, mask).

Signals are decimated 5000 -> 1250 samples (effective 125 Hz) to keep CPU
training tractable; this is documented as an architectural limitation.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402
from ecgdet.atrial_residual import extract_atrial_residual  # noqa: E402
from ecgdet.lead_quality_audit import build_lead_mask  # noqa: E402
from ecgdet.dataset_harmonization import decimate_signal  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "data" / "processed" / "af_afl_stage3611"
OUT.mkdir(parents=True, exist_ok=True)
FS, N_RR, DEC = 500.0, 40, 4
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def one(sig: np.ndarray):
    peaks, _ = multi_lead_qrs(sig, FS, LEADS)
    rr = np.diff(peaks) / FS if len(peaks) >= 2 else np.array([0.8])
    rr = np.clip(rr, 0.1, 2.0)[:N_RR]
    padded = np.zeros(N_RR, dtype=np.float32)
    padded[:len(rr)] = rr
    mask = build_lead_mask(sig, FS, LEADS)
    res = extract_atrial_residual(sig, FS, LEADS, method="median", qrs_peaks=peaks)
    atrial = np.asarray(res["residual"], dtype=np.float32)
    # Stage 3.6.15: antialias-filtered resampling, NOT naive x[::DEC].
    atrial = np.nan_to_num(decimate_signal(atrial, FS, FS / DEC))[:12, :1250]
    raw = decimate_signal(sig, FS, FS / DEC)[:12, :1250]
    return padded, mask, atrial, raw


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    out = Parallel(n_jobs=8, batch_size=32, verbose=5)(
        delayed(one)(np.asarray(X[i], dtype=np.float32)) for i in range(len(X))
    )
    rr = np.asarray([o[0] for o in out], dtype=np.float32)
    mask = np.asarray([o[1] for o in out], dtype=bool)
    atrial = np.asarray([o[2] for o in out], dtype=np.float32)
    raw = np.asarray([o[3] for o in out], dtype=np.float32)
    np.savez_compressed(OUT / "inputs.npz", rr=rr, mask=mask, atrial=atrial, raw=raw)
    stats = {
        "n": len(rr), "rr_shape": rr.shape, "atrial_shape": atrial.shape,
        "decimation": DEC, "effective_fs": FS / DEC,
        "atrial_nan_frac": float(np.isnan(atrial).mean()),
        "mean_usable_leads": float(mask.mean(axis=1).mean()),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
