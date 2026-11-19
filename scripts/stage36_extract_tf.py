"""Stage 3.6.5 — extract STFT + organization features on atrial residual."""
from __future__ import annotations
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.time_frequency import stft_atrial_features
from ecgdet.organization import atrial_organization_features

D = ROOT / "data" / "processed" / "af_afl_stage36"
D34 = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"


def one(i, X):
    rec = np.asarray(X[i], dtype=float)  # (leads, samples)
    # lead II of the median residual when available, else raw lead II bandpassed
    lead = rec[1] if rec.shape[0] > 1 else rec[0]
    out = {}
    for k, v in stft_atrial_features(lead, 500.0).items():
        out["stft_" + k] = v
    for k, v in atrial_organization_features(lead, 500.0).items():
        out["org_" + k] = v
    return out


def main():
    X = np.load(D / "features_atrial_residual.npy", mmap_mode="r")
    # features matrix has 431 columns; spectral residual of lead II is at index of 'median_II_*'
    names = pd.read_csv(D / "feature_names_atrial_residual.csv")["name"].tolist()
    # Instead of reconstructing residual from features, use raw ECG lead II directly
    Xr = np.load(D34 / "X.npy", mmap_mode="r")
    res = Parallel(n_jobs=8, verbose=0, batch_size=16)(delayed(one)(i, Xr) for i in range(len(Xr)))
    keys = sorted({k for d in res for k in d})
    M = np.asarray([[float(d.get(k, np.nan)) for k in keys] for d in res], dtype=np.float32)
    np.save(D / "features_temporal_organization.npy", M)
    pd.Series(keys).to_csv(D / "feature_names_temporal_organization.csv", index=False, header=["name"])
    print("TF features", M.shape, flush=True)


if __name__ == "__main__":
    main()
