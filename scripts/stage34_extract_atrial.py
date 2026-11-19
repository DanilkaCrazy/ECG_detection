"""Stage 3.4 — extract atrial_activity_v2 features for af_afl_stage34."""

from __future__ import annotations

import os, sys, subprocess, hashlib, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage34"


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["AF_AFL_OUT"] = "af_afl_stage34"
    env["AF_AFL_FEAT"] = "atrial34"
    env["AF_AFL_JOBS"] = "8"
    # custom one-off extraction: feature function selected via env in a small runner script
    code = '''
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
'''
    p = ROOT / "scripts" / "_stage34_extract_atrial.py"
    p.write_text(code, encoding="utf-8")
    rc = subprocess.run([sys.executable, str(p)], env=env, cwd=ROOT)
    if rc.returncode:
        raise RuntimeError("atrial extraction failed")
    h = hashlib.sha256((D / "features_atrial34.npy").read_bytes()).hexdigest()
    info = {"rows": len(np.load(D / "features_atrial34.npy", mmap_mode="r")),
            "sha256": h, "file": "features_atrial34.npy"}
    MET.mkdir(parents=True, exist_ok=True)
    (MET / "atrial_feature_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
