#!/usr/bin/env python
"""Audit all local ECG datasets and write reproducible manifests.

Outputs
-------
reports/dataset_integrity_report.json  aggregated summary
reports/manifests/<dataset>_manifest.csv per-record status table
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecgdet import audit, config  # noqa: E402

MANIFEST_DIR = config.REPORTS_ROOT / "manifests"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def summarize(df: pd.DataFrame, name: str) -> dict:
    counts = df["status"].value_counts().to_dict() if "status" in df else {}
    summary = {
        "dataset": name,
        "n_records": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "usable": int((df["status"] == "ok").sum()) if "status" in df else 0,
    }
    if "checksum_ok" in df:
        summary["checksum_ok"] = int(df["checksum_ok"].sum())
        summary["checksum_fail"] = int((~df["checksum_ok"].sum() if False else (~df["checksum_ok"]).sum()))
    if "shape_ok" in df:
        summary["shape_ok"] = int(df["shape_ok"].sum())
        summary["shape_fail"] = int((~df["shape_ok"]).sum())
    if "has_signal_file" in df:
        summary["with_signal_file"] = int(df["has_signal_file"].sum())
        summary["without_signal_file"] = int((~df["has_signal_file"]).sum())
    if "saturation_fraction" in df:
        sat = df["saturation_fraction"].dropna()
        summary["saturation_fraction"] = {
            "mean": float(sat.mean()) if len(sat) else None,
            "p99": float(sat.quantile(0.99)) if len(sat) else None,
            "max": float(sat.max()) if len(sat) else None,
        }
    return summary


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(config.PROJECT_ROOT),
        "datasets": [],
    }

    # ------------------------------------------------------------------ chapman
    print("=== Chapman-Shaoxing ===", flush=True)
    chap = audit.audit_chapman(verbose=True)
    chap_manifest = MANIFEST_DIR / "chapman_manifest.csv"
    chap.to_csv(chap_manifest, index=False)

    orphan_mats = chap.attrs.get("orphan_mats", [])
    orphan_headers = chap.attrs.get("orphan_headers", [])
    summary = summarize(chap, "chapman")
    summary["n_header_files"] = chap.attrs.get("n_headers")
    summary["n_mat_files"] = chap.attrs.get("n_mats")
    summary["orphan_mat_without_hea"] = len(orphan_mats)
    summary["orphan_hea_without_mat"] = len(orphan_headers)
    summary["orphan_mat_examples"] = orphan_mats[:20]
    summary["orphan_hea_examples"] = orphan_headers[:20]
    report["datasets"].append(summary)
    print(chap["status"].value_counts().to_string())

    # ------------------------------------------------------------------- ptb-xl
    print("\n=== PTB-XL ===", flush=True)
    try:
        ptb = audit.audit_ptbxl()
        ptb.drop(columns=["scp_codes"], errors="ignore").to_csv(
            MANIFEST_DIR / "ptbxl_manifest.csv", index=False
        )
        summary = summarize(ptb, "ptbxl")
        summary["n_patients"] = int(ptb["patient_id"].nunique())
        summary["fold_counts"] = {
            str(k): int(v) for k, v in ptb["strat_fold"].value_counts().items()
        }
        report["datasets"].append(summary)
        print(ptb["status"].value_counts().to_string())
    except Exception as exc:  # noqa: BLE001
        report["datasets"].append({"dataset": "ptbxl", "error": f"{type(exc).__name__}: {exc}"})
        print(f"  ptbxl audit failed: {exc}")

    # ------------------------------------------------------------------ mit-bih
    print("\n=== MIT-BIH ===", flush=True)
    mit = audit.audit_mitbih()
    mit.to_csv(MANIFEST_DIR / "mitbih_manifest.csv", index=False)
    summary = summarize(mit, "mitbih")
    report["datasets"].append(summary)
    print(mit["status"].value_counts().to_string())

    # ------------------------------------------------------------------- incart
    print("\n=== INCART ===", flush=True)
    inc = audit.audit_incart()
    inc.to_csv(MANIFEST_DIR / "incart_manifest.csv", index=False)
    summary = summarize(inc, "incart")
    summary["n_patients"] = int(inc["patient_id"].nunique()) if "patient_id" in inc else None
    report["datasets"].append(summary)
    print(inc["status"].value_counts().to_string())

    out = config.REPORTS_ROOT / "dataset_integrity_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Wrote manifests to {MANIFEST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
