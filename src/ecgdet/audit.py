"""Dataset integrity audits.

The audit answers one question per file: *can this record be used as-is?*
Every record receives a ``status`` string plus boolean flags so that the
downstream pipeline can apply a documented, reproducible exclusion policy
instead of silently dropping data.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .wfdb_io import (
    Header,
    parse_header,
    parse_header_lenient,
    read_mat_signal,
    referenced_files,
    wfdb_checksum,
)

__all__ = [
    "chapman_status",
    "audit_chapman",
    "audit_ptbxl",
    "audit_mitbih",
    "audit_incart",
    "sha256_of",
]


# --------------------------------------------------------------------- utilities


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _empty_row(record: str) -> dict:
    return {
        "record": record,
        "status": "ok",
        "has_header": False,
        "has_signal_file": False,
        "signal_file": None,
        "shape_ok": False,
        "checksum_ok": False,
        "init_value_ok": False,
        "n_sig": 0,
        "n_samp": 0,
        "fs": np.nan,
        "saturation_fraction": np.nan,
        "nonfinite_fraction": np.nan,
        "reason": "",
    }


# ----------------------------------------------------------------------- chapman


def chapman_status(header_path: str | Path, deep: bool = True) -> dict:
    """Audit a single Chapman record (``.hea`` + referenced ``.mat``)."""
    header_path = Path(header_path)
    row = _empty_row(header_path.stem)
    row["has_header"] = True

    try:
        header, note = parse_header_lenient(header_path)
    except Exception as exc:  # noqa: BLE001 - we want the reason, not a crash
        row["status"] = "unparsable_header"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row
    if header is None:
        row["status"] = "unparsable_header"
        row["reason"] = note
        return row
    row["repair"] = note
    row["repaired"] = not note.startswith("ok")

    row.update(
        n_sig=header.n_sig,
        n_samp=header.n_samp,
        fs=header.fs,
        record=header.record_name,
    )
    row["dx_codes"] = ",".join(header.dx_codes)
    row["age"] = header.comments.get("Age", "")
    row["sex"] = header.comments.get("Sex", "")

    files = referenced_files(header)
    existing = [f for f in files if f.exists()]
    row["signal_file"] = files[0].name if files else None

    if not existing:
        row["status"] = "header_without_signal_file"
        row["reason"] = f"missing: {[f.name for f in files]}"
        return row

    row["has_signal_file"] = True
    signal_path = existing[0]

    if not deep:
        row["status"] = "ok"
        return row

    if signal_path.suffix.lower() != ".mat":
        row["status"] = "unexpected_signal_format"
        row["reason"] = f"signal file is not .mat: {signal_path.name}"
        return row

    try:
        val = read_mat_signal(signal_path)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "unreadable_signal_file"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row

    # ---- shape agreement
    shape_ok = val.shape[0] == header.n_sig and (
        header.n_samp == 0 or val.shape[1] == header.n_samp
    )
    row["shape_ok"] = bool(shape_ok)
    row["mat_shape"] = f"{val.shape[0]}x{val.shape[1]}"

    # ---- WFDB checksum agreement (per channel, modulo 2**16)
    checks = [s.checksum for s in header.signals]
    if all(c is not None for c in checks) and val.shape[0] == len(checks):
        computed = [wfdb_checksum(val[i]) for i in range(val.shape[0])]
        checksum_ok = all(
            (c - int(e)) % 65536 == 0 for c, e in zip(computed, checks)
        )
        row["checksum_ok"] = bool(checksum_ok)
        row["n_checksum_mismatch"] = int(
            sum(int((c - int(e)) % 65536 != 0) for c, e in zip(computed, checks))
        )
    else:
        row["checksum_ok"] = False
        row["n_checksum_mismatch"] = -1

    # ---- first-sample (init value) agreement
    init_ok = True
    for i, sig in enumerate(header.signals):
        if i < val.shape[0] and val.shape[1] > 0:
            if int(val[i, 0]) != int(sig.init_value):
                init_ok = False
                break
    row["init_value_ok"] = bool(init_ok)

    # ---- saturation / finiteness
    if np.issubdtype(val.dtype, np.integer):
        sat = float(np.mean(np.abs(val.astype(np.int64)) >= 32767))
    else:
        sat = float(np.mean(~np.isfinite(val)))
    row["saturation_fraction"] = sat
    row["nonfinite_fraction"] = float(np.mean(~np.isfinite(val)))

    # ---- final status
    if not shape_ok:
        row["status"] = "shape_mismatch"
        row["reason"] = f"mat {val.shape} != header ({header.n_sig}x{header.n_samp})"
    elif not row["checksum_ok"]:
        row["status"] = "checksum_mismatch"
        row["reason"] = f"{row['n_checksum_mismatch']} channel(s) differ from header"
    elif not init_ok:
        row["status"] = "init_value_mismatch"
        row["reason"] = "first sample differs from header init value"
    else:
        row["status"] = "ok"
    return row


def _chapman_worker(args) -> dict:
    header_path, deep = args
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return chapman_status(header_path, deep=deep)
        except Exception as exc:  # noqa: BLE001 - worker must never die
            last_exc = exc
            if isinstance(exc, ImportError):
                # transient DLL/memory pressure from many concurrent workers
                time.sleep(0.5 * (attempt + 1))
                continue
            break
    return {
        "record": Path(header_path).stem,
        "status": "audit_error",
        "reason": f"{type(last_exc).__name__}: {last_exc}",
        "has_header": True,
    }


def audit_chapman(
    root: str | Path | None = None,
    deep: bool = True,
    workers: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Audit the whole Chapman-Shaoxing collection.

    Returns a frame with one row per ``.hea`` plus extra rows for orphaned
    ``.mat`` files that have no matching header.
    """
    root = Path(root) if root else config.dataset_root("chapman") / config.CHAPMAN_SUBDIR
    records_root = root / config.CHAPMAN_RECORDS_ROOT

    headers = sorted(records_root.rglob("*.hea"))
    mats = sorted(records_root.rglob("*.mat"))
    if verbose:
        print(f"[chapman] headers={len(headers)} mats={len(mats)}")

    header_stems = {p.stem for p in headers}
    mat_stems = {p.stem for p in mats}

    workers = workers or min(8, max(1, (os.cpu_count() or 4) - 2))
    tasks = [(str(p), deep) for p in headers]
    rows: list[dict] = []
    if deep:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_chapman_worker, tasks, chunksize=64)):
                rows.append(row)
                if verbose and (i + 1) % 5000 == 0:
                    print(f"[chapman] audited {i + 1}/{len(tasks)}")
    else:
        rows = [_chapman_worker(t) for t in tasks]

    df = pd.DataFrame(rows)
    df["folder"] = [str(Path(p).parent.relative_to(records_root)) for p in headers]

    # ---- orphans: .mat without .hea
    orphan_mats = sorted(mat_stems - header_stems)
    orphan_hea = sorted(header_stems - mat_stems)
    if verbose:
        print(
            f"[chapman] orphan .mat (no .hea): {len(orphan_mats)}; "
            f".hea without .mat: {len(orphan_hea)}"
        )
    df.attrs["orphan_mats"] = orphan_mats
    df.attrs["orphan_headers"] = orphan_hea
    df.attrs["n_mats"] = len(mats)
    df.attrs["n_headers"] = len(headers)
    return df


# ------------------------------------------------------------------------ ptb-xl


def audit_ptbxl(root: str | Path | None = None) -> pd.DataFrame:
    """Audit PTB-XL: metadata rows vs. signal files actually present."""
    root = Path(root) if root else config.dataset_root("ptbxl")
    base = root / "physionet.org/files/ptb-xl/1.0.3"
    meta_path = base / config.PTBXL_FILENAME
    scp_path = base / config.PTBXL_SCP_FILENAME

    meta = pd.read_csv(meta_path, index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)

    rows = []
    for ecg_id, row in meta.iterrows():
        filename = row["filename_hr"]
        hea = base / f"{filename}.hea"
        dat = base / f"{filename}.dat"
        lr = row.get("filename_lr")
        rows.append(
            {
                "record": str(ecg_id),
                "patient_id": row.get("patient_id"),
                "filename_he": filename,
                "filename_lr": lr,
                "has_header": hea.exists(),
                "has_signal_file": dat.exists(),
                "n_samp": row.get("n_samples"),
                "fs": 500.0,
                "baseline_drift": row.get("baseline_drift"),
                "static_noise": row.get("static_noise"),
                "burst_noise": row.get("burst_noise"),
                "strat_fold": row.get("strat_fold"),
                "scp_codes": row.get("scp_codes"),
                "age": row.get("age"),
                "sex": row.get("sex"),
                "device": row.get("device"),
            }
        )
    df = pd.DataFrame(rows)
    df["status"] = np.where(
        df["has_header"] & df["has_signal_file"], "ok", "missing_files"
    )
    df.attrs["scp_statements"] = scp
    df.attrs["metadata"] = meta
    return df


# ---------------------------------------------------------------------- mit-bih


def audit_mitbih(root: str | Path | None = None) -> pd.DataFrame:
    """Audit MIT-BIH Arrhythmia Database records and their annotations."""
    root = Path(root) if root else config.dataset_root("mitbih")
    base = root / "physionet.org/files/mitdb/1.0.0"

    rows = []
    for hea in sorted(base.glob("*.hea")):
        rec = hea.stem
        try:
            header = parse_header(hea)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "record": rec,
                    "status": "unparsable_header",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        dat = base / f"{rec}.dat"
        atr = base / f"{rec}.atr"
        row = {
            "record": rec,
            "has_header": True,
            "has_signal_file": dat.exists(),
            "has_annotation": atr.exists(),
            "n_sig": header.n_sig,
            "n_samp": header.n_samp,
            "fs": header.fs,
            "lead_names": ",".join(header.lead_names),
            "status": "ok" if dat.exists() else "missing_signal_file",
        }
        if header.comments.get("Age"):
            row["age"] = header.comments.get("Age")
        if header.comments.get("Sex"):
            row["sex"] = header.comments.get("Sex")
        if header.comments.get("Dx"):
            row["dx"] = header.comments.get("Dx")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------- incart


def audit_incart(root: str | Path | None = None) -> pd.DataFrame:
    """Audit INCART 12-lead database."""
    root = Path(root) if root else config.dataset_root("incart")
    base = root / "physionet.org/files/incartdb/1.0.0"

    diag_path = base / "files-patients-diagnoses.txt"
    diagnoses: dict[str, str] = {}
    if diag_path.exists():
        for line in diag_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                diagnoses[parts[0]] = " ".join(parts[1:])

    rows = []
    for hea in sorted(base.glob("*.hea")):
        rec = hea.stem
        try:
            header = parse_header(hea)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {"record": rec, "status": "unparsable_header", "reason": str(exc)}
            )
            continue
        dat = base / f"{rec}.dat"
        atr = base / f"{rec}.atr"
        rows.append(
            {
                "record": rec,
                "patient_id": rec[1:],
                "has_header": True,
                "has_signal_file": dat.exists(),
                "has_annotation": atr.exists(),
                "n_sig": header.n_sig,
                "n_samp": header.n_samp,
                "fs": header.fs,
                "lead_names": ",".join(header.lead_names),
                "diagnosis": diagnoses.get(rec[1:], ""),
                "status": "ok" if dat.exists() else "missing_signal_file",
            }
        )
    return pd.DataFrame(rows)
