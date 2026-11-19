
import os, sys, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.environ["PYTHONPATH"].split(os.pathsep)[0])
from ecgdet.atrial_signal_analysis import (multi_lead_qrs, qrs_cancel_compare,
    spectral_atrial, temporal_atrial, multilead_coherence, lead_index_map)
from ecgdet import config
D = config.PROCESSED_ROOT / "af_afl_stage34"
X = np.load(D / "X.npy", mmap_mode="r")
meta = pd.read_csv(D / "meta.csv")
LEADS = ["I","II","III","aVR","aVL","aVF"] + [f"V{i}" for i in range(1,7)]

def one(i):
    s = np.asarray(X[i]).T   # (samples, leads)
    peaks, q = multi_lead_qrs(s, 500.0, LEADS)
    out = {}
    out.update({f"qrs_{k}": float(v) for k, v in q.items() if isinstance(v, (int,float))})
    out["qrs_n_consensus"] = float(len(peaks))
    cancel = qrs_cancel_compare(s, 500.0, LEADS, peaks)
    for method, d in cancel.items():
        out[f"cancel_{method}_resid_energy"] = float(d["qrs_residual_energy"])
        out[f"cancel_{method}_preservation"] = float(d["preservation"])
    li = lead_index_map(LEADS)
    for l in ("II","III","aVF","V1"):
        if l not in li: continue
        x = s[:, li[l]]
        for k, v in spectral_atrial(x, 500.0).items():
            out[f"spec_{k}_{l}"] = v
        for k, v in temporal_atrial(x, 500.0, peaks).items():
            out[f"temp_{k}_{l}"] = v
    for k, v in multilead_coherence(s, 500.0, LEADS).items():
        out[k] = v
    return out

res = Parallel(n_jobs=8, verbose=5, batch_size=8)(delayed(one)(i) for i in range(len(X)))
keys = sorted({k for d in res for k in d.keys()})
M = np.array([[float(d.get(k, np.nan)) for k in keys] for d in res], dtype=np.float32)
np.save(D / "features_atrial_advanced.npy", M)
pd.Series(keys).to_csv(D / "feature_names_atrial_advanced.csv", index=False, header=["name"])
assert len(M) == len(meta)
print("atrial advanced", M.shape)
