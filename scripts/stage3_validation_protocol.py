"""Stage 3.0 — rebuild leak-free grouped splits and experiment sets A-E.

- drops duplicate signals (keeps first occurrence);
- uses StratifiedGroupKFold on patient_key for ALL sources;
- marks cpsc2018 (no AFL) as pretrain-only (not binary training);
- writes manifest_stage3.csv, split_stage3.csv and a protocol report.
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

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_merged_v2"
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
MET.mkdir(parents=True, exist_ok=True)
OUT = config.PROCESSED_ROOT / "af_afl_stage3"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MET / "manifest.csv")
    n0 = len(manifest)
    manifest["orig_idx"] = np.arange(n0)
    manifest = manifest.drop_duplicates("signal_hash", keep="first").reset_index(drop=True)
    print(f"deduplicated: {n0} -> {len(manifest)}", flush=True)

    # binary-eligible = both classes present in source
    src_counts = manifest.groupby("source")["label"].agg(["sum", "count"])
    src_counts["n_neg"] = src_counts["count"] - src_counts["sum"]
    binary_ok = src_counts[(src_counts["sum"] > 0) & (src_counts["n_neg"] > 0)].index.tolist()
    pretrain_only = [s for s in manifest.source.unique() if s not in binary_ok]
    print("binary sources:", binary_ok, "pretrain-only:", pretrain_only, flush=True)

    # grouped stratified split: 3 folds -> train / val / test
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    split = np.full(len(manifest), "unassigned", dtype=object)
    y = manifest["label"].to_numpy()
    groups = manifest["patient_key"].to_numpy()
    # split within each source to preserve domain balance
    for src, sub in manifest.groupby("source"):
        idx = sub.index.to_numpy()
        # cpsc2018: no AFL -> pretrain only
        if src not in binary_ok:
            split[idx] = "pretrain_only"
            continue
        folds = list(sgkf.split(idx, y[idx], groups=groups[idx]))
        val_idx = idx[folds[0][1]]
        test_idx = idx[folds[1][1]]
        train_idx = np.setdiff1d(idx, np.concatenate([val_idx, test_idx]))
        split[val_idx] = "val"
        split[test_idx] = "test"
        split[train_idx] = "train"
    # ensure every source has all three splits
    for src in binary_ok:
        sub = manifest[manifest.source == src]
        print(f"  {src}: " + ", ".join(f"{s}={int((split[sub.index] == s).sum())}"
                                       for s in ("train", "val", "test")), flush=True)

    manifest["split_stage3"] = split
    manifest["binary_eligible"] = manifest.source.isin(binary_ok)
    manifest.to_csv(MET / "manifest_stage3.csv", index=False)
    pd.DataFrame({"idx": np.arange(len(manifest)), "split": split,
                  "source": manifest.source.values, "label": y,
                  "patient_key": groups}).to_csv(OUT / "split.csv", index=False)

    # experiment sets A-E
    sets = {
        "A_chapman_only": ["chapman"],
        "B_ptbxl_only": ["ptbxl"],
        "C_cpsc_extra_only": ["cpsc2018_extra"],
        "D_chapman_ptbxl": ["chapman", "ptbxl"],
        "E_all_binary": binary_ok,
    }
    rows = []
    for name, sources in sets.items():
        sel = manifest[manifest.source.isin(sources)]
        rows.append({"set": name, "sources": ",".join(sources), "n": len(sel),
                     "n_afib": int(sel.label.sum()), "n_afl": int((sel.label == 0).sum()),
                     "train": int((sel.split_stage3 == "train").sum()),
                     "val": int((sel.split_stage3 == "val").sum()),
                     "test": int((sel.split_stage3 == "test").sum())})
    sets_df = pd.DataFrame(rows)
    sets_df.to_csv(MET / "experiment_sets.csv", index=False)

    lines = [
        "# Stage 3.0 — протокол валидации и Experiment Sets", "",
        f"После удаления дубликатов: {len(manifest)} записей (было {n0}).", "",
        "## Разбиение", "",
        "Использован `StratifiedGroupKFold(3)` **внутри каждого источника** "
        "с группировкой по `patient_key` (Chapman: record_id, PTB-XL: patient_id, SHDB: запись). "
        "Одно разбиение используется как val, второе как test, остальное как train.", "",
        "Источники с обеими классами (binary-eligible): " + ", ".join(binary_ok) + ".",
        "Источники только для pretraining AFIB vs others: " + (", ".join(pretrain_only) or "нет") + ".", "",
        "## Experiment sets", "",
        sets_df.to_markdown(index=False) if False else sets_df.to_string(index=False), "",
        " CPSC2018 (1221 записей, 0 AFL) исключён из бинарного обучения — "
        "используется только для pretraining AFIB vs others.", "",
        "## Правила", "",
        "- Тест не используется для подбора гиперпараметров, порога или калибровки;",
        "- калибровка и порог — только на val;",
        "- все метрики — с bootstrap CI и парным сравнением на одинаковых записях;",
        "- для PTB-XL проверено отсутствие пересечения пациентов val/test.",
    ]
    (ROOT / "reports" / "AF_AFL_STAGE3_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:25]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
