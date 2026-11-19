"""Extract 10-second AFIB/AFL windows from the long SHDB-AF recordings.

Strict labelling: a window is accepted only if it lies fully inside one
annotated rhythm interval (>= margin from both boundaries). Windows from the
same source recording never cross splits (group split by record).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
SRC = config.RAW_ROOT / "shdb-af"
OUT = config.PROCESSED_ROOT / "shdb_af_windows"
FS, LEN, LEADS = 500, 5000, 12
MAX_PER_REC = {"AFIB": 30, "AFL": 15}   # windows per recording per rhythm
MARGIN_S = 5


def rhythm_intervals(rec: str):
    ann = wfdb.rdann(str(SRC / rec), "atr")
    fs = ann.fs or 200
    events = []
    for s, a in zip(ann.sample, ann.aux_note):
        a = (a or "").strip()
        if "AFIB" in a:
            events.append((int(s), 1))
        elif "AFL" in a:
            events.append((int(s), 0))
        elif a in ("(N", "(N\0", "(NSR", "(SR"):
            events.append((int(s), -1))
    events.sort()
    intervals = []
    for i, (s, lab) in enumerate(events):
        if lab < 0:
            continue
        end = events[i + 1][0] if i + 1 < len(events) else None
        if end is not None and (end - s) / fs >= (LEN / FS + 2 * MARGIN_S):
            intervals.append((s, end, lab))
    return intervals, fs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = sorted({p.stem for p in SRC.glob("*.atr")})
    print(f"SHDB-AF records: {len(recs)}", flush=True)
    Xs, ys, meta = [], [], []
    for rec in recs:
        intervals, fs = rhythm_intervals(rec)
        counts = {"AFIB": 0, "AFL": 0}
        for s0, s1, lab in intervals:
            rhythm = "AFIB" if lab == 1 else "AFL"
            if counts[rhythm] >= MAX_PER_REC[rhythm]:
                continue
            lo = s0 + int(MARGIN_S * fs)
            hi = s1 - int((LEN / FS + MARGIN_S) * fs)
            if hi <= lo:
                continue
            # evenly spaced windows inside the interval
            n = min(MAX_PER_REC[rhythm] - counts[rhythm], max(1, int((hi - lo) / (2 * LEN / FS * fs))))
            starts = np.linspace(lo, hi, n + 2)[1:-1].astype(int) if n > 1 else [int((lo + hi) / 2)]
            for st in starts:
                try:
                    r = wfdb.rdrecord(str(SRC / rec), sampfrom=st, sampto=st + int(LEN / FS * fs))
                    sig = np.asarray(r.p_signal, dtype=np.float32)
                except Exception:
                    continue
                if sig.shape[0] < 100:
                    continue
                if abs(fs - FS) > 1e-6:
                    g = math.gcd(FS, int(fs))
                    sig = resample_poly(sig, FS // g, int(fs) // g, axis=0).astype(np.float32)
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
                    continue
                Xs.append(np.ascontiguousarray(sig.T, dtype=np.float32))
                ys.append(lab)
                meta.append({"source": "shdb-af", "record_id": rec, "patient_id": rec,
                             "rhythm": rhythm, "start_sample": int(st)})
                counts[rhythm] += 1
        if (recs.index(rec) + 1) % 20 == 0:
            print(f"  {recs.index(rec)+1}/{len(recs)} records, windows={len(ys)}", flush=True)

    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=np.int64)
    print(f"Windows: {len(X)} (AFIB {int(y.sum())}, AFL {int((y==0).sum())})", flush=True)
    mm = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32, shape=X.shape)
    mm[:] = X; mm.flush()
    np.save(OUT / "y.npy", y)
    m = pd_meta = __import__("pandas").DataFrame(meta)
    m.to_csv(OUT / "meta.csv", index=False)
    grp = m["record_id"].values
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr, tmp = next(gss.split(np.zeros((len(X), 1)), y, groups=grp))
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        d = np.lib.format.open_memmap(OUT / f"X_{name}.npy", mode="w+", dtype=np.float32,
                                      shape=(len(ids), LEADS, LEN))
        for k, i in enumerate(ids):
            d[k] = X[i]
        d.flush()
        np.save(OUT / f"y_{name}.npy", y[ids])
    __import__("pandas").DataFrame({
        "split": ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te),
        "idx": np.concatenate([tr, va, te]),
    }).to_csv(OUT / "split.csv", index=False)
    stats = {"n": len(X), "n_afib": int(y.sum()), "n_afl": int((y == 0).sum()),
             "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "n_records": len(recs), "max_per_rec": MAX_PER_REC,
             "geometry": {"fs": FS, "n_samp": LEN, "n_leads": LEADS}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
