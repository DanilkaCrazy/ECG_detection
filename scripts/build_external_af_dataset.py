"""Build a unified EXTERNAL AFIB/AFL dataset from PTB-XL and CPSC2018.

Only strict labels are kept:
- PTB-XL: AFIB in scp_codes and no AFLT; AFLT in codes and no AFIB.
- CPSC2018 / CPSC2018-extra: SNOMED 164889003 (AFIB) without 164890007 (AFL)
  and vice versa; records with both codes are dropped as ambiguous.

All signals are resampled to 500 Hz, trimmed/padded to 5000 samples (10 s),
12 leads, NaN -> 0, float32 — identical geometry to the Chapman dataset.
"""

from __future__ import annotations

import ast
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
RAW = config.RAW_ROOT
OUT = config.PROCESSED_ROOT / "af_external"
AFIB, AFL = "164889003", "164890007"
FS, LEN, LEADS = 500.0, 5000, 12


def load_one(rec: dict):
    import wfdb
    from scipy.signal import resample_poly
    try:
        r = wfdb.rdrecord(str(rec["path"]))
        sig = np.asarray(r.p_signal, dtype=np.float32)
        fs = float(r.fs)
        if abs(fs - FS) > 1e-6:
            g = math.gcd(int(FS), int(fs))
            sig = resample_poly(sig, int(FS) // g, int(fs) // g, axis=0).astype(np.float32)
        if sig.shape[0] < LEN:
            sig = np.pad(sig, ((0, LEN - sig.shape[0]), (0, 0)), mode="constant")
        else:
            sig = sig[:LEN]
        if sig.shape[1] < LEADS:
            sig = np.pad(sig, ((0, 0), (0, LEADS - sig.shape[1])), mode="constant")
        else:
            sig = sig[:, :LEADS]
        sig = np.nan_to_num(sig)
        if np.std(sig) < 1e-3 or np.max(np.abs(sig)) > 100:
            return None
        return np.ascontiguousarray(sig.T, dtype=np.float32)
    except Exception:
        return None


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    # PTB-XL
    ptb = RAW / "ptbxl" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
    db = pd.read_csv(ptb / "ptbxl_database.csv")
    for _, r in db.iterrows():
        try:
            codes = ast.literal_eval(str(r["scp_codes"]))
        except Exception:
            continue
        if not isinstance(codes, dict):
            continue
        has_afib, has_afl = "AFIB" in codes, "AFLT" in codes
        if has_afib == has_afl:  # neither or both -> ambiguous
            continue
        rows.append({
            "source": "ptbxl", "record_id": str(r.name),
            "patient_id": r.get("patient_id"), "rhythm": "AFIB" if has_afib else "AFL",
            "path": str(ptb / str(r["filename_hr"])),
        })
    # CPSC2018 + extra
    for dn, tag in (("cpsc2018", "cpsc2018"), ("cpsc2018_extra", "cpsc2018_extra")):
        d = RAW / dn / "data"
        for f in d.rglob("*.hea"):
            txt = f.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"#\s*Dx:\s*(.+)", txt)
            if not m:
                continue
            codes = [c for c in re.split(r"[,;\s]+", m.group(1).strip()) if c]
            ha, hf = AFIB in codes, AFL in codes
            if ha == hf:
                continue
            rows.append({
                "source": tag, "record_id": f.stem, "patient_id": None,
                "rhythm": "AFIB" if ha else "AFL", "path": str(f.with_suffix("")),
            })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    print(f"Strict external candidates: {len(rows)} "
          f"(AFIB={sum(r['rhythm']=='AFIB' for r in rows)}, AFL={sum(r['rhythm']=='AFL' for r in rows)})", flush=True)
    sigs = Parallel(n_jobs=8, verbose=5, batch_size=16)(delayed(load_one)(r) for r in rows)
    keep = [(r, s) for r, s in zip(rows, sigs) if s is not None]
    X = np.array([s for _, s in keep], dtype=np.float32)
    meta = pd.DataFrame([r for r, _ in keep])
    y = np.where(meta["rhythm"] == "AFIB", 1, 0).astype(int)
    print(f"Loaded {len(X)}: AFIB={int(y.sum())}, AFL={int((y==0).sum())}", flush=True)

    # patient/group-aware split: PTB-XL patient_id, CPSC record as patient
    grp = meta["patient_id"].fillna(meta["source"] + "_" + meta["record_id"]).astype(str).values
    mm = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32, shape=X.shape)
    mm[:] = X; mm.flush()
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr, tmp = next(gss.split(np.zeros((len(X), 1)), y, groups=grp))
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        dst = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+",
                                        dtype=np.float32, shape=(len(ids), LEADS, LEN))
        for k, i in enumerate(ids):
            dst[k] = X[i]
        dst.flush()
        np.save(OUT / f"y_{name}.npy", y[ids])
    np.save(OUT / "y.npy", y)
    meta.to_csv(OUT / "meta.csv", index=False)
    pd.DataFrame({"split": ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te),
                  "idx": np.concatenate([tr, va, te])}).to_csv(OUT / "split.csv", index=False)
    stats = {"n": len(X), "n_afib": int(y.sum()), "n_afl": int((y == 0).sum()),
             "by_source": meta["source"].value_counts().to_dict(),
             "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "geometry": {"fs": FS, "n_samp": LEN, "n_leads": LEADS},
             "label_rule": "strict primary rhythm code; ambiguous pairs dropped"}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
