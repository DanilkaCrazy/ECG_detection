"""Stage 3.1 — build a single audited dataset from manifest_final.csv.

Usage of source arrays:
  chapman  -> data/processed/af_afl_strict/X.npy (by record_id)
  external -> data/processed/af_external/X.npy   (by source+record_id)
SHDB-AF is EXCLUDED (lead_mask all False) and CPSC2018 (no AFL) is excluded
from binary training. All rows are taken in manifest_final order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
OUT = config.PROCESSED_ROOT / "af_afl_stage31"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MET / "manifest_final.csv")
    man = man[man.split != "pretrain_only"].reset_index(drop=True)
    man = man[man.source != "shdb-af"].reset_index(drop=True)

    ch = pd.read_csv(config.PROCESSED_ROOT / "af_afl_strict" / "meta.csv")
    ex = pd.read_csv(config.PROCESSED_ROOT / "af_external" / "meta.csv")
    CX = np.load(config.PROCESSED_ROOT / "af_afl_strict" / "X.npy", mmap_mode="r")
    EX = np.load(config.PROCESSED_ROOT / "af_external" / "X.npy", mmap_mode="r")
    cpos = {str(r): i for i, r in enumerate(ch.record_id)}
    epos = {f"{s}:{r}": i for i, (s, r) in enumerate(zip(ex.source, ex.record_id))}

    rows, misses = [], 0
    for _, r in man.iterrows():
        if r.source == "chapman":
            i = cpos.get(str(r.record_id))
            src = CX if i is not None else None
        else:
            i = epos.get(f"{r.source}:{r.record_id}")
            src = EX if i is not None else None
        if src is None or i is None:
            misses += 1
            rows.append(None)
        else:
            rows.append(np.asarray(src[i], dtype=np.float32))
    if misses:
        bad = man[[s is None for s in rows]][["source", "record_id"]]
        bad.to_csv(OUT / "excluded_records.csv", index=False)
        raise RuntimeError(f"{misses} manifest records not found in source arrays; "
                           f"see excluded_records.csv")
    keep = [i for i, s in enumerate(rows) if s is not None]
    man = man.iloc[keep].reset_index(drop=True)
    X = np.array([rows[i] for i in keep], dtype=np.float32)

    mm = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32, shape=X.shape)
    mm[:] = X; mm.flush()
    np.save(OUT / "y.npy", man.label.to_numpy(dtype=int))
    man.to_csv(OUT / "meta.csv", index=False)
    pd.DataFrame({"idx": np.arange(len(man)), "split": man.split.values,
                  "source": man.source.values, "label": man.label.values,
                  "patient_key": man.patient_key.values}).to_csv(OUT / "split.csv", index=False)
    h_src = hashlib.sha256(pd.read_csv(MET / "manifest_final.csv").to_csv(index=False).encode()).hexdigest()
    h_final = hashlib.sha256(man.to_csv(index=False).encode()).hexdigest()
    stats = {"n": len(man), "by_source": man.source.value_counts().to_dict(),
             "by_split": man.split.value_counts().to_dict(),
             "n_afib": int((man.label == 1).sum()), "n_afl": int((man.label == 0).sum()),
             "manifest_final_sha256_source": h_src,
             "manifest_final_sha256_actual": h_final,
             "manifest_final_sha256": h_final,
             "excluded": ["shdb-af (no canonical leads)", "cpsc2018 (pretrain-only)"]}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
