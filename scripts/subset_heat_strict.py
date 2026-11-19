"""Subset precomputed HEAT features for the strict dataset by record_id."""

from pathlib import Path
import numpy as np
import pandas as pd
from ecgdet import config

SRC = config.PROCESSED_ROOT / "af_afl_full10s_dedup"
OUT = config.PROCESSED_ROOT / "af_afl_strict"
m_src = pd.read_csv(SRC / "meta.csv")
m_out = pd.read_csv(OUT / "meta.csv")
pos = {rid: i for i, rid in enumerate(m_src["record_id"])}
idx = np.array([pos[r] for r in m_out["record_id"]], dtype=int)
H = np.load(SRC / "features_heat.npy", mmap_mode="r")
np.save(OUT / "features_heat.npy", np.asarray(H[idx]))
np.save(OUT / "features_combined_heat.npy",
        np.concatenate([np.load(OUT / "features_combined.npy"), np.asarray(H[idx])], axis=1))
print("strict HEAT subset:", len(idx), flush=True)
