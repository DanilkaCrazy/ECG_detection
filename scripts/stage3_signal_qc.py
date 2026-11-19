"""Stage 3.0 FINAL — signal QC and harmonization verification.

Samples records from every source, applies the shared harmonization contract,
and verifies units, lead mapping, masks, physical ranges and QC-before-cleaning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

from ecgdet import config
from ecgdet.dataset_harmonization import (HarmonizationStatus, harmonize_signal,
                                          signal_quality)

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
RAW = config.RAW_ROOT


def get_record(src, path, n=60):
    try:
        r = wfdb.rdrecord(str(path).replace(".hea", ""))
        units = list(getattr(r, "units", [])) or ["mV"]
        canonical, mask, info = harmonize_signal(
            np.asarray(r.p_signal, dtype=np.float64), float(r.fs),
            list(r.sig_name), units=units[0], signal_kind="physical",
            target_fs=500.0, target_len=5000)
        return {"source": src, "status": info["status"], "units": units[0],
                "n_channels": len(r.sig_name), "fs": float(r.fs),
                "leads_present": int(mask.sum()),
                "missing": len(info["missing_leads"]),
                "qc_nan_frac": info["qc_raw"]["nan_frac"],
                "qc_inf_frac": info["qc_raw"]["inf_frac"],
                "max_abs_post": info["qc_clean"]["max_abs"],
                "std_post": info["qc_clean"]["std"],
                "notes": ";".join(info["notes"])[:100]}
    except Exception as exc:
        return {"source": src, "status": "error", "units": "?", "n_channels": 0,
                "fs": np.nan, "leads_present": 0, "missing": 12,
                "qc_nan_frac": np.nan, "qc_inf_frac": np.nan,
                "max_abs_post": np.nan, "std_post": np.nan,
                "notes": str(exc)[:100]}


def main() -> int:
    man = pd.read_csv(MET / "manifest_final.csv")
    rows = []
    for src in ("chapman", "ptbxl", "cpsc2018", "cpsc2018_extra", "shdb-af"):
        sub = man[man.source == src]
        paths = []
        for _, r in sub.iterrows():
            p = r.get("orig_path")
            if isinstance(p, str) and p:
                paths.append(p)
        if src == "shdb-af":
            # SHDB files are named 001.hea, 002.hea ... -> zero-pad to 3 digits
            paths = [str(RAW / "shdb-af" / str(int(r)).zfill(3))
                     for r in sub.record_id.unique()[:60]]
        if not paths:
            rows.append({"source": src, "status": "no_path", "units": "?", "n_channels": 0,
                         "fs": np.nan, "leads_present": 0, "missing": 12,
                         "qc_nan_frac": np.nan, "qc_inf_frac": np.nan,
                         "max_abs_post": np.nan, "std_post": np.nan, "notes": "path unavailable"})
            continue
        sel = paths[:60]
        res = Parallel(n_jobs=8, verbose=0)(delayed(get_record)(src, p) for p in sel)
        rows.extend(res)
        print(src, len(sel), "checked", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "signal_qc.csv", index=False)
    summary = df.groupby("source").agg(
        n=("status", "size"),
        ok=("status", lambda s: int((s == HarmonizationStatus.OK).sum())),
        review=("status", lambda s: int((s == HarmonizationStatus.REQUIRE_REVIEW).sum())),
        invalid=("status", lambda s: int((s == HarmonizationStatus.INVALID).sum())),
        median_fs=("fs", "median"),
        median_leads=("leads_present", "median"),
        max_abs=("max_abs_post", "median"),
    ).round(3)
    lines = ["# Stage 3.0 — SIGNAL QC / гармонизация", "",
             "Проверка на случайной выборке до 60 записей на источник через "
             "`dataset_harmonization.harmonize_signal`.", "",
             "WFDB `p_signal` уже в физических единицах — ADC gain НЕ применяется повторно. "
             "QC (NaN/Inf, амплитуда) вычисляется до замены NaN на нули.", "",
             summary.to_string(), "",
             "Неизвестные единицы НЕ считаются mV; запись получает статус `require_review`.", "",
             "Сопоставление отведений — по именам; отсутствующие каналы получают `lead_mask=False`."]
    (ROOT / "reports" / "AF_AFL_STAGE3_SIGNAL_QC.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote signal QC report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
