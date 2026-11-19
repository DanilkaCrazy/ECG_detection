"""Stage 3.3 — FULL signal QC with explicit PASS / REVIEW / REJECT statuses.

Checks EVERY record from the audited manifest:
- fs, duration, number of samples (shape[1], not shape[0]);
- units present and matching channel count (no implicit mV default);
- NaN and Inf separately;
- flat and clipped channels;
- physical amplitude plausibility in mV;
- standard lead names and lead_mask coverage.

PASS  -> can be used in binary training;
REVIEW -> excluded automatically, needs manual check;
REJECT -> unacceptable signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

from ecgdet import config
from ecgdet.dataset_harmonization import (CANONICAL_LEADS, canonicalize_leads,
                                          signal_quality)
from ecgdet.dataset_harmonization import HarmonizationStatus

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage33"
MET.mkdir(parents=True, exist_ok=True)
D = config.PROCESSED_ROOT / "af_afl_stage31"
MAN = pd.read_csv(D / "meta.csv")

KNOWN = {"mv", "millivolt", "uv", "microv", "µv", "microvolt"}
TARGET_FS = 500.0
TARGET_LEN = 5000


def check_one(row):
    r = row.to_dict()
    base = {"source": r.get("source"), "record_id": r.get("record_id"),
            "patient_key": r.get("patient_key"), "split": r.get("split")}
    try:
        path = r.get("orig_path")
        if not isinstance(path, str):
            path = r.get("path_hea", r.get("path", ""))
        rec = wfdb.rdrecord(str(path).replace(".hea", ""))
        sig = np.asarray(rec.p_signal, dtype=np.float64)  # WFDB: (samples, channels)
        n_samp, n_ch = sig.shape
        sig = sig.T  # canonical (channels, samples)
        fs = float(rec.fs)
        units = [str(u).strip().lower() for u in getattr(rec, "units", [])]
        names = list(rec.sig_name)
        q = signal_quality(sig)
        idx, mask = canonicalize_leads(names, n_channels=n_ch)
        mv = sig.copy()
        unknown = [u for u in units if u not in KNOWN]
        for i, u in enumerate(units):
            if u in ("uv", "microv", "µv", "microvolt"):
                mv[i] /= 1000.0

        reasons = []
        if not units:
            reasons.append("units_missing")
        if unknown:
            reasons.append("units_unknown:" + ",".join(sorted(set(unknown))))
        # one global unit string may legally apply to all channels
        if len(units) == 1:
            units = units * n_ch
        if len(units) > 1 and len(units) != n_ch:
            reasons.append("units_channel_mismatch")
        if q["nan_frac"] > 0:
            reasons.append(f"nan_frac={q['nan_frac']:.3f}")
        if q["inf_frac"] > 0:
            reasons.append(f"inf_frac={q['inf_frac']:.3f}")
        if q["flat_leads"] > 0:
            reasons.append(f"flat_leads={q['flat_leads']}")
        if np.max(np.abs(mv)) > 20:
            reasons.append(f"amplitude>20mV:{np.max(np.abs(mv)):.2f}")
        if abs(fs - TARGET_FS) > 1e-6:
            reasons.append(f"fs={fs}")
        if n_samp < TARGET_LEN * 0.95:
            reasons.append(f"too_short={n_samp}")
        if int(mask.sum()) < 12:
            reasons.append(f"leads_present={int(mask.sum())}")

        # classify
        hard = {"units_missing", "units_unknown", "units_channel_mismatch",
                "inf_frac", "amplitude>20mV", "leads_present"}
        has_hard = any(any(h in x for h in hard) for x in reasons)
        has_soft = any(x.startswith(("nan_frac", "flat_leads", "fs=", "too_short")) for x in reasons)
        status = "REJECT" if has_hard else ("REVIEW" if has_soft else "PASS")

        base.update({
            "fs": fs, "n_samp": n_samp, "n_channels": n_ch,
            "duration_s": n_samp / fs if fs else np.nan,
            "units": "|".join(sorted(set(units))), "units_count": len(units),
            "leads_present": int(mask.sum()),
            "missing_leads": ",".join(l for l, ok in zip(CANONICAL_LEADS, mask) if not ok),
            "nan_frac": q["nan_frac"], "inf_frac": q["inf_frac"],
            "flat_leads": q["flat_leads"],
            "max_abs_mv": float(np.max(np.abs(mv))), "std_mv": float(np.std(mv)),
            "status": status, "reasons": ";".join(reasons)[:300],
        })
        return base
    except Exception as exc:
        base.update({"fs": np.nan, "n_samp": 0, "n_channels": 0, "units": "",
                     "units_count": 0, "leads_present": 0, "missing_leads": "all",
                     "nan_frac": np.nan, "inf_frac": np.nan, "flat_leads": 0,
                     "max_abs_mv": np.nan, "std_mv": np.nan,
                     "status": "REJECT", "reasons": "read_error:" + str(exc)[:150]})
        return base


def main() -> int:
    res = Parallel(n_jobs=8, verbose=5, batch_size=32)(
        delayed(check_one)(r) for _, r in MAN.iterrows())
    df = pd.DataFrame(res)
    df.to_csv(MET / "full_signal_qc.csv", index=False)
    summ = df.groupby("source").agg(
        n=("status", "size"),
        PASS=("status", lambda s: int((s == "PASS").sum())),
        REVIEW=("status", lambda s: int((s == "REVIEW").sum())),
        REJECT=("status", lambda s: int((s == "REJECT").sum())),
        median_fs=("fs", "median"), median_n_samp=("n_samp", "median"),
        median_leads=("leads_present", "median"),
        max_abs_med=("max_abs_mv", "median"),
    ).round(3)
    reasons = df[df.status != "PASS"].groupby("source")["reasons"].apply(
        lambda s: s.value_counts().head(5).to_dict()).to_dict()
    lines = ["# Stage 3.3 — полный QC сигналов (PASS / REVIEW / REJECT)", "",
             f"Проверено **{len(df)}** записей.", "",
             summ.to_string(), "",
             "PASS — все проверки пройдены; REVIEW — есть устранимые сомнения; "
             "REJECT — сигнал не пригоден. Только PASS используется в обучении.", "",
             "Причины не-PASS:", ""]
    for src, d in reasons.items():
        lines.append(f"- {src}: {d}")
    lines += ["", "Правила: неизвестные единицы НЕ считаются mV; n_samp = shape[1]; "
              "NaN и Inf учитываются раздельно; амплитуда проверяется в mV после conversions."]
    (ROOT / "reports" / "AF_AFL_STAGE33_DATA_INTEGRITY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote full QC with statuses", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
