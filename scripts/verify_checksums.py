#!/usr/bin/env python
"""Verify every local file of a PhysioNet collection against its published SHA256SUMS.

This is the strongest available check for "missing / extra / corrupted" files:
the sums are the ones published by PhysioNet, so a mismatch means the local
copy is not byte-identical to the official release.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecgdet import config  # noqa: E402


def read_sums(sums_path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    with sums_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, _, name = line.partition(" ")
            sums[name.strip()] = digest.strip()
    return sums


def _hash_file(args) -> tuple[str, str | None, str]:
    root, name, expected = args
    path = Path(root) / name
    if not path.exists():
        return name, None, "missing_locally"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got == expected:
        return name, got, "ok"
    return name, got, "hash_mismatch"


def verify(root: Path, sums_path: Path, workers: int | None = None, verbose: bool = True) -> pd.DataFrame:
    sums = read_sums(sums_path)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    tasks = [(str(root), name, digest) for name, digest in sums.items()]
    rows: list[tuple[str, str | None, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_hash_file, tasks, chunksize=32)):
            rows.append(row)
            if verbose and (i + 1) % 10000 == 0:
                print(f"[sha256] {i + 1}/{len(tasks)}")
    df = pd.DataFrame(rows, columns=["file", "local_sha256", "status"])

    local_files = {
        str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()
    }
    listed = set(sums)
    df_extra = pd.DataFrame(
        {
            "file": sorted(local_files - listed),
            "local_sha256": [None] * len(local_files - listed),
            "status": ["not_in_official_manifest"] * len(local_files - listed),
        }
    )
    df = pd.concat([df, df_extra], ignore_index=True)
    return df


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "chapman"
    if target == "chapman":
        root = config.dataset_root("chapman") / config.CHAPMAN_SUBDIR
        sums = root / "SHA256SUMS.txt"
    elif target == "ptbxl":
        root = config.dataset_root("ptbxl") / "physionet.org/files/ptb-xl/1.0.3"
        sums = root / "SHA256SUMS.txt"
    elif target == "mitbih":
        root = config.dataset_root("mitbih") / "physionet.org/files/mitdb/1.0.0"
        sums = root / "SHA256SUMS.txt"
    elif target == "incart":
        root = config.dataset_root("incart") / "physionet.org/files/incartdb/1.0.0"
        sums = root / "SHA256SUMS.txt"
    else:
        raise SystemExit(f"unknown target: {target}")

    if not sums.exists():
        raise SystemExit(f"no SHA256SUMS.txt under {root}")

    print(f"[sha256] verifying {root} against {sums.name}")
    df = verify(root, sums)
    out_csv = config.REPORTS_ROOT / "manifests" / f"{target}_sha256.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    counts = df["status"].value_counts().to_dict()
    print(json.dumps(counts, indent=2))

    bad = df[df["status"] != "ok"]
    if len(bad):
        print("\nNon-OK files:")
        print(bad.head(50).to_string())

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": target,
        "root": str(root),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "problems": bad["file"].tolist()[:200],
    }
    out = config.REPORTS_ROOT / f"{target}_sha256_report.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_csv} and {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
