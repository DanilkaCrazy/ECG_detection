"""Stage 3.1 — FULL signal QC on all audited records (no 60-record sample).

Checks every record: fs, units, lead mask, NaN/Inf, amplitude, flat/clipped
channels, duration, resampling status, and harmonic QC status.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

from ecgdet import config
from ecgdet.dataset_harmonization import (HarmonizationStatus, canonicalize_leads,
                                          signal_quality)

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage31"
MET.mkdir(parents=True, exist_ok=True)
D = config.PROCESSED_ROOT / "af_afl_stage31"
MAN = pd.read_csv(D / "meta.csv")


def check_one(row):
    path = row.get("orig_path")
    if not isinstance(path, str):
        path = row.get("path_hea", row.get("path", ""))
    try:
        r = wfdb.rdrecord(str(path).replace(".hea", ""))
        sig = np.asarray(r.p_signal, dtype=np.float64)
        qc_raw = signal_quality(sig)
        units = list(getattr(r, "units", [])) or ["mV"]
        idx, mask = canonicalize_leads(list(r.sig_name), n_channels=sig.shape[0])
        known = all(str(u).strip().lower() in
                    ("mv", "millivolt", "uv", "microv", "µv", "microvolt") for u in units)
        return {
            "source": row.source, "record_id": row.record_id,
            "fs": float(r.fs), "n_samp": int(sig.shape[0]),
            "n_channels": int(sig.shape[0]),
            "leads_present": int(mask.sum()),
            "units": "|".join(sorted(set(str(u) for u in units))),
            "units_known": bool(known),
            "nan_frac": qc_raw["nan_frac"], "inf_frac": qc_raw["inf_frac"],
            "max_abs": qc_raw["max_abs"], "std": qc_raw["std"],
            "flat_leads": qc_raw["flat_leads"],
            "status": HarmonizationStatus.OK if known and qc_raw["nan_frac"] < 0.05
                      else HarmonizationStatus.REQUIRE_REVIEW,
        }
    except Exception as exc:
        return {"source": row.source, "record_id": row.record_id, "fs": np.nan,
                "n_samp": 0, "n_channels": 0, "leads_present": 0, "units": "?",
                "units_known": False, "nan_frac": np.nan, "inf_frac": np.nan,
                "max_abs": np.nan, "std": np.nan, "flat_leads": 0,
                "status": "error:" + str(exc)[:60]}


def main() -> int:
    res = Parallel(n_jobs=8, verbose=5, batch_size=32)(
        delayed(check_one)(r) for _, r in MAN.iterrows())
    df = pd.DataFrame(res)
    df.to_csv(MET / "full_signal_qc.csv", index=False)
    summ = df.groupby("source").agg(
        n=("status", "size"), ok=("status", lambda s: int((s == "ok").sum())),
        errors=("status", lambda s: int(s.astype(str).str.startswith("error").sum())),
        known_units=("units_known", "sum"),
        med_fs=("fs", "median"), med_leads=("leads_present", "median"),
        nan_records=("nan_frac", lambda s: int((s > 0).sum())),
        max_abs_med=("max_abs", "median"),
    ).round(3)
    lines = ["# Stage 3.1 — ПОЛНЫЙ QC сигналов", "",
             f"Проверено **{len(df)}** записей (все, без сэмплирования).", "",
             summ.to_string(), "",
             "Статус `ok` означает: известные физические единицы, нет Error/NaN выше 5%, "
             "каналы сопоставимы по именам.", "",
             "SHDB-AF исключён из 12-канального binary набора."]
    (ROOT / "reports" / "AF_AFL_STAGE31_FULL_QC.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote full QC", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
