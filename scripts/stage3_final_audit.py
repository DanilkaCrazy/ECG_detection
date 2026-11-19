"""Stage 3.0 FINAL: rebuild unified manifest from corrected sources.

Sources:
- Chapman strict (primary #Dx), from af_afl_strict;
- external PTB-XL / CPSC-extra (new harmonized loader), from af_external;
- SHDB-AF windows, from shdb_af_windows.
CPSC2018 (no AFL) is marked pretrain_only.

Actions: deduplicate by signal_hash BEFORE splitting, build group-aware
StratifiedGroupKFold(3) splits within each source, verify zero patient overlap
and zero duplicate signals across splits, write final manifest + audit reports.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ecgdet import config
from ecgdet.dataset_harmonization import signal_hash

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
PROC = config.PROCESSED_ROOT


def load_source(name, path, meta_file, y_file):
    m = pd.read_csv(path / meta_file)
    y = np.load(path / y_file)
    if "label" in m.columns:
        y = m["label"].to_numpy()
    if "signal_hash" not in m.columns:
        X = np.load(path / "X.npy", mmap_mode="r")
        m["signal_hash"] = [signal_hash(np.asarray(X[i], dtype=np.float32)) for i in range(len(X))]
    out = pd.DataFrame({
        "source": name if "source" not in m.columns else m["source"],
        "record_id": m["record_id"].astype(str),
        "patient_id": m.get("patient_id"),
        "rhythm": m.get("rhythm"),
        "label": y,
        "signal_hash": m["signal_hash"],
        "orig_path": m.get("path_hea", m.get("path", None)),
    })
    return out


def main() -> int:
    ch = load_source("chapman", PROC / "af_afl_strict", "meta.csv", "y.npy")
    ex = load_source("external", PROC / "af_external", "meta.csv", "y.npy")
    sh = load_source("shdb-af", PROC / "shdb_af_windows", "meta.csv", "y.npy")
    man = pd.concat([ch, ex, sh], ignore_index=True)
    n_before = len(man)

    # label conflicts among identical signals
    conflicts = int((man.groupby("signal_hash")["label"].nunique() > 1).sum())
    dup = man.duplicated("signal_hash", keep="first").to_numpy()
    man = man[~dup].reset_index(drop=True)

    def pat(r):
        if pd.notna(r.get("patient_id")):
            return f"{r['source']}:{int(r['patient_id'])}"
        return f"{r['source']}:{r['record_id']}"
    man["patient_key"] = man.apply(pat, axis=1)

    # CPSC2018 without AFL -> pretrain only
    counts = man.groupby("source")["label"].agg(["sum", "count"])
    counts["n_neg"] = counts["count"] - counts["sum"]
    binary_ok = counts[(counts["sum"] > 0) & (counts["n_neg"] > 0)].index.tolist()

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    split = np.full(len(man), "pretrain_only", dtype=object)
    for src in binary_ok:
        sub = man[man.source == src]
        idx = sub.index.to_numpy()
        folds = list(sgkf.split(idx, sub.label.to_numpy(), groups=sub.patient_key.to_numpy()))
        val, test = idx[folds[0][1]], idx[folds[1][1]]
        train = np.setdiff1d(idx, np.concatenate([val, test]))
        split[val] = "val"; split[test] = "test"; split[train] = "train"
    man["split"] = split

    # verification
    lines = ["# Stage 3.0 — FINAL DATA AUDIT", "",
             f"Записей до дедупликации: **{n_before}**, после: **{len(man)}** "
             f"(удалено {n_before-len(man)}).", "",
             f"Конфликтов разметки у одинаковых сигналов: **{conflicts}**.", "",
             "## patient overlap", "",
             "| source | records train/val/test | patients train/val/test | intersect tr-val, tr-te, val-te |",
             "|---|---:|---:|---:|"]
    ok = True
    for src in sorted(man.source.unique()):
        sub = man[man.source == src]
        ss = {s: set(sub.loc[sub.split == s, "patient_key"]) for s in ("train", "val", "test")}
        a, b, c = len(ss["train"] & ss["val"]), len(ss["train"] & ss["test"]), len(ss["val"] & ss["test"])
        if a + b + c:
            ok = False
        rec = [int((sub.split == s).sum()) for s in ("train", "val", "test")]
        lines.append(f"| {src} | {rec[0]}/{rec[1]}/{rec[2]} | "
                     f"{len(ss['train'])}/{len(ss['val'])}/{len(ss['test'])} | {a}/{b}/{c} |")
    hcross = int(man.groupby("signal_hash")["split"].nunique().gt(1).sum())
    lines += ["", f"Сигналов с хэшем в нескольких сплитах: **{hcross}**.", "",
              f"Статус: **{'PASS' if ok and hcross == 0 else 'FAIL'}**.", "",
              "Источники без обоих классов (`pretrain_only`): " +
              ", ".join([s for s in man.source.unique() if s not in binary_ok]) + ".", "",
              "Ограничение: Chapman не имеет patient_id; group = record_id "
              "(одна запись = один пациент). PTB-XL group = patient_id.", ""]
    (ROOT / "reports" / "AF_AFL_STAGE3_FINAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    man.to_csv(MET / "manifest_final.csv", index=False)

    # split validation report
    vlines = ["# Stage 3.0 — SPLIT VALIDATION", "",
              "Групповое разделение: `StratifiedGroupKFold(3, shuffle=True, random_state=42)` "
              "внутри каждого источника. Один фолд → val, второй → test, остальное → train.", "",
              "\n".join(lines[8:]), "",
              "Дубликаты удалены до разделения; Thест не участвует в подборе модели.", ""]
    (ROOT / "reports" / "AF_AFL_STAGE3_SPLIT_VALIDATION.md").write_text("\n".join(vlines).replace("Thест", "Тест"), encoding="utf-8")
    print("Wrote final audit reports", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
