"""Stage 3.6.10 — build the unified five-group feature matrix.

Groups B/C are loaded from the already-validated Stage 3.6 arrays (no
recomputation, no leakage).  Groups A/D/E are computed from raw ECG with the
audit-corrected, mask-aware helpers.
"""
from __future__ import annotations

import json
import sys
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs  # noqa: E402
from ecgdet.rhythm_features import rhythm_features, FEATURE_NAMES as A_NAMES  # noqa: E402
from ecgdet.morphology_features import morphology_features, FEATURE_NAMES as D_NAMES  # noqa: E402
from ecgdet.interlead_features import interlead_features, FEATURE_NAMES as E_NAMES  # noqa: E402
from ecgdet.lead_quality_audit import build_lead_mask  # noqa: E402
from ecgdet.atrial_features import select_atrial_features  # noqa: E402
from ecgdet.time_frequency_features import (  # noqa: E402
    select_tf_features, select_organization_features,
)

D = ROOT / "data" / "processed" / "af_afl_stage34"
S36 = ROOT / "data" / "processed" / "af_afl_stage36"
OUT = ROOT / "data" / "processed" / "af_afl_stage3610"
OUT.mkdir(parents=True, exist_ok=True)
FS = 500.0
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"] + [f"V{i}" for i in range(1, 7)]


def one_record(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = build_lead_mask(sig, FS, LEADS)
    peaks, _ = multi_lead_qrs(sig, FS, LEADS)
    rr = np.diff(peaks) / FS if len(peaks) >= 2 else np.array([])
    ra = rhythm_features(rr)
    # Morphology on the two atrial-friendly leads; fall back to any usable lead.
    morph_leads = [i for i in (1, 6) if mask[i]] or [i for i in range(12) if mask[i]][:1]
    md = []
    for li in morph_leads:
        f = morphology_features(sig[li], peaks, FS)
        md += [f[k] for k in D_NAMES]
    if len(morph_leads) == 1:
        md += [0.0] * len(D_NAMES)
    ie = interlead_features(sig, FS, LEADS, lead_mask=mask)
    a = np.array([ra[k] for k in A_NAMES], dtype=np.float32)
    d = np.array(md[:2 * len(D_NAMES)], dtype=np.float32)
    e = np.array([ie[k] for k in E_NAMES], dtype=np.float32)
    return a, d, e


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    results = Parallel(n_jobs=8, batch_size=32, verbose=5)(
        delayed(one_record)(np.asarray(X[i], dtype=np.float32)) for i in range(len(X))
    )
    A = np.asarray([r[0] for r in results], dtype=np.float32)
    Dm = np.asarray([r[1] for r in results], dtype=np.float32)
    E = np.asarray([r[2] for r in results], dtype=np.float32)

    bnames = pd.read_csv(S36 / "feature_names_atrial_residual.csv")["name"].astype(str).tolist()
    Bmat = np.load(S36 / "features_atrial_residual.npy", mmap_mode="r")
    bidx = select_atrial_features(bnames)
    B = np.asarray(Bmat[:, bidx], dtype=np.float32)

    cnames = pd.read_csv(S36 / "feature_names_temporal_organization.csv")["name"].astype(str).tolist()
    Cmat = np.load(S36 / "features_temporal_organization.npy", mmap_mode="r")
    cidx = select_tf_features(cnames)
    C = np.asarray(Cmat[:, cidx], dtype=np.float32)
    orgidx = select_organization_features(cnames)
    Dorg = np.asarray(Cmat[:, orgidx], dtype=np.float32)
    Dfull = np.column_stack([Dm, Dorg])

    names = ([f"A_{n}" for n in A_NAMES]
             + [f"B_{bnames[i]}" for i in bidx]
             + [f"C_{cnames[i]}" for i in cidx]
             + [f"D_{n}_lead{li}" for li in (1, 6) for n in D_NAMES]
             + [f"D_{cnames[i]}" for i in orgidx]
             + [f"E_{n}" for n in E_NAMES])
    groups = (["A"] * A.shape[1] + ["B"] * B.shape[1] + ["C"] * C.shape[1]
              + ["D"] * Dfull.shape[1] + ["E"] * E.shape[1])
    unified = np.column_stack([A, B, C, Dfull, E]).astype(np.float32)
    assert unified.shape[1] == len(names) == len(groups)

    np.save(OUT / "X.npy", unified)
    pd.DataFrame({"name": names, "group": groups}).to_csv(OUT / "feature_names.csv", index=False)
    meta.to_csv(OUT / "meta.csv", index=False)
    pd.read_csv(D / "split.csv").to_csv(OUT / "split.csv", index=False)
    stats = {
        "n_records": int(unified.shape[0]), "n_features": int(unified.shape[1]),
        "by_group": {g: int(sum(x == g for x in groups)) for g in "ABCDE"},
        "nan_frac": float(np.isnan(unified).mean()),
        "source": "A/D/E computed from raw X; B/C loaded from Stage 3.6 arrays",
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
