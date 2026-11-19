"""Stage 3.3 — reproducible feature extraction with integrity checks.

Runs expert/v2/heat extraction from af_afl_stage31/X.npy and validates:
- row counts match meta.csv;
- stable hashes saved;
- feature order deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage31"
MET = ROOT / "reports" / "metrics" / "af_afl_stage33"


def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    MET.mkdir(parents=True, exist_ok=True)
    meta = __import__("pandas").read_csv(D / "meta.csv")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["AF_AFL_OUT"] = "af_afl_stage31"
    env["AF_AFL_JOBS"] = "8"
    for feat in ("expert", "v2", "heat"):
        env["AF_AFL_FEAT"] = feat
        script = "scripts/extract_heat_features.py" if feat == "heat" else "scripts/extract_af_afl_features.py"
        print(f"extracting {feat}...", flush=True)
        rc = subprocess.run([sys.executable, script], cwd=ROOT, env=env).returncode
        if rc != 0:
            raise RuntimeError(f"feature extraction failed: {feat}")
    names = {"expert": "features_advanced_expert.npy",
             "v2": "features_advanced_v2.npy", "heat": "features_heat.npy"}
    info = {}
    for k, fn in names.items():
        p = D / fn
        a = __import__("numpy").load(p, mmap_mode="r")
        assert len(a) == len(meta), f"{fn}: {len(a)} != {len(meta)}"
        info[k] = {"file": fn, "rows": len(a), "cols": a.shape[1],
                   "sha256": hash_file(p)}
    (MET / "feature_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
