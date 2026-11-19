"""Explore annotation availability for QRS detection benchmark."""
import sys, pathlib, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import wfdb
import numpy as np
from collections import Counter
from pathlib import Path
RAW = Path("data/raw")

for ds in ["edb", "mitbih", "ptbdb", "qtdb", "svdb", "vfdb", "cpsc2018_extra", "ptbxl"]:
    fs = sorted(RAW.glob(f"{ds}/**/*.atr"))
    if not fs:
        continue
    r = wfdb.rdrecord(str(fs[0].with_suffix("")))
    ann = wfdb.rdann(str(fs[0].with_suffix("")), "atr")
    c = Counter(ann.symbol)
    print(f"{ds:12} {len(fs):4} recs  fs={r.fs:5} n_ann={len(ann.symbol):6} codes={dict(c.most_common(8))}")
