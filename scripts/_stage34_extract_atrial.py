
import os, sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.environ["PYTHONPATH"].split(os.pathsep)[0])
from ecgdet.atrial_activity_v2 import atrial_features
from ecgdet import config
D = config.PROCESSED_ROOT / os.environ["AF_AFL_OUT"]
X = np.load(D / "X.npy", mmap_mode="r")
meta = pd.read_csv(D / "meta.csv")

def one(i):
    lead_names = ["I","II","III","aVR","aVL","aVF"] + [f"V{k}" for k in range(1,7)]
    return atrial_features(np.asarray(X[i]).T, 500.0, lead_names)

res = Parallel(n_jobs=8, verbose=5, batch_size=8)(delayed(one)(i) for i in range(len(X)))
keys = sorted({k for d in res for k in d.keys()})
M = np.array([[float(d.get(k, np.nan)) for k in keys] for d in res], dtype=np.float32)
np.save(D / "features_atrial34.npy", M)
pd.Series(keys).to_csv(D / "feature_names_atrial34.csv", index=False, header=["name"])
assert len(M) == len(meta), (len(M), len(meta))
print("atrial34 features", M.shape)
