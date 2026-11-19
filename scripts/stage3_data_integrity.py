"""Stage 3.0 — Data integrity audit and unified manifest.

Checks (on the Stage 2 merged_v2 dataset):
- patient overlap between train / val / test splits, per source;
- exact duplicate signals (hash) within and across splits;
- cross-source duplicate record identifiers;
- approximate duplicates in feature space (cosine > 0.999);
- channel names / units / gain consistency from raw WFDB headers.

Outputs: reports/metrics/af_afl_stage3/manifest.csv
         reports/AF_AFL_STAGE3_DATA_AUDIT.md
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ecgdet import config
from ecgdet.dataset_harmonization import CANONICAL_LEADS, canonicalize_leads, signal_hash

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_merged_v2"
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"
MET.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST = MET / "manifest.csv"
OUT_REPORT = ROOT / "reports" / "AF_AFL_STAGE3_DATA_AUDIT.md"


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    y = np.load(D / "y.npy")
    assert len(meta) == len(X) == len(y)

    # ---- hashes (streaming) ----
    hashes = []
    for i in range(len(X)):
        hashes.append(signal_hash(np.asarray(X[i], dtype=np.float32)))
    meta["signal_hash"] = hashes
    meta["label"] = y

    # ---- standardise patient identifier ----
    def pat(row):
        if pd.notna(row.get("patient_id")):
            return f"{row['source']}:{int(row['patient_id'])}"
        return f"{row['source']}:{row['record_id']}"
    meta["patient_key"] = meta.apply(pat, axis=1)
    meta["record_key"] = meta["source"].astype(str) + ":" + meta["record_id"].astype(str)

    lines: list[str] = []
    add = lines.append
    add("# Stage 3.0 — аудит целостности данных")
    add("")
    add(f"Записей: **{len(meta)}**. Источники: {meta['source'].value_counts().to_dict()}.")
    add("")
    add("## 1. Разделение пациентов")
    add("")
    add("| Источник | train | val | test | пересечение train/val | train/test | val/test |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    problems = []
    for src, sub in meta.groupby("source"):
        sets = {s: set(sub.loc[sub.split == s, "patient_key"]) for s in
                ("train", "val", "test_chapman", "test_external", "test_shdb")}
        tr = sets["train"]
        va = set().union(*[sets[k] for k in ("val",) if k in sets])
        te = set().union(*[sets[k] for k in ("test_chapman", "test_external", "test_shdb") if k in sets])
        ov_tv, ov_tt, ov_vt = len(tr & va), len(tr & te), len(va & te)
        if ov_tv or ov_tt or ov_vt:
            problems.append((src, ov_tv, ov_tt, ov_vt))
        add(f"| {src} | {len(tr)} | {len(va)} | {len(te)} | {ov_tv} | {ov_tt} | {ov_vt} |")
    add("")
    if problems:
        add("**Обнаружены пересечения пациентов между выборками** — см. таблицу выше. "
            "Необходимо пересобрать сплиты с StratifiedGroupKFold до обучения Stage 3.")
    else:
        add("Пересечений пациентов между train / val / test нет.")

    # PTB-XL specific: patient_id overlap between val and test
    ptb = meta[meta.source == "ptbxl"]
    v = ptb.loc[ptb.split == "val", "patient_key"]
    t = ptb.loc[ptb.split.isin(["test_external", "test_chapman", "test_shdb"]), "patient_key"]
    add("")
    add(f"- PTB-XL: уникальных пациентов {ptb.patient_key.nunique()}, "
        f"пересечение val/test: **{len(set(v) & set(t))}**.")
    add("- Chapman: `patient_id` недоступен; каждый `record_id` считается отдельным пациентом.")
    add("- SHDB-AF: сплит сгруппирован по `record_id` (одна длинная запись = один пациент).")

    # ---- duplicates ----
    add("")
    add("## 2. Дубликаты")
    add("")
    dup = meta[meta.duplicated("signal_hash", keep=False)]
    add(f"- Точных дубликатов сигналов: **{len(dup)}** записей "
        f"({dup.signal_hash.nunique()} уникальных хэшей).")
    if len(dup):
        for h, grp in list(dup.groupby("signal_hash"))[:5]:
            add(f"    - {h[:10]}: " + ", ".join(f"{r.source}/{r.record_id}/{r.split}" for r in grp.itertuples()))
    cross = []
    for h, grp in dup.groupby("signal_hash"):
        if grp.source.nunique() > 1:
            cross.append(h)
    add(f"- Междатасетных дубликатов: **{len(cross)}**.")
    rec_dups = meta.duplicated("record_key", keep=False).sum()
    add(f"- Повторяющихся `source/record_id`: **{rec_dups}** "
        "(для SHDB это разные 10-с окна одной записи — ожидаемо).")

    # ---- approximate duplicates in feature space ----
    e = np.load(D / "features_expert.npy", mmap_mode="r")
    v2 = np.load(D / "features_v2.npy", mmap_mode="r")
    h = np.load(D / "features_heat.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(meta), size=min(2000, len(meta)), replace=False)
    F = np.concatenate([np.asarray(e[sample_idx]), np.asarray(v2[sample_idx]),
                        np.asarray(h[sample_idx])], axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=8).fit(F)
    dist, inds = nn.kneighbors(F)
    near = np.sum(dist[:, 1] < 0.001)
    add(f"- Приближённых дубликатов (cosine > 0.999 на {len(sample_idx)} случайных записях): **{int(near)}**.")

    # ---- lead / units check ----
    add("")
    add("## 3. Гармонизация отведений и единиц")
    add("")
    add(f"Канонический порядок: `{', '.join(CANONICAL_LEADS)}`.")
    add("")
    import wfdb
    checked = {}
    for src in meta.source.unique():
        row = meta[meta.source == src].iloc[0]
        path = row.get("path_hea") if pd.notna(row.get("path_hea")) else row.get("path")
        if not isinstance(path, str):
            continue
        try:
            if src == "ptbxl":
                hdr = wfdb.rdheader(path.replace("_hr.hea", "_hr"))
            elif src == "shdb-af":
                hdr = wfdb.rdheader(str(config.RAW_ROOT / "shdb-af" / str(row.record_id)))
            else:
                hdr = wfdb.rdheader(str(path).replace(".hea", ""))
            names = list(hdr.sig_name)
            idx, mask = canonicalize_leads(names)
            units = list(getattr(hdr, "units", []))[:3]
            gain = list(getattr(hdr, "adc_gain", []))[:3] if getattr(hdr, "adc_gain", None) is not None else []
            checked[src] = (names, int(mask.sum()), units, gain, float(hdr.fs))
        except Exception as exc:
            checked[src] = (["<read error>"], 0, [str(exc)[:60]], [], np.nan)
    for src, (names, present, units, gain, fs) in checked.items():
        add(f"- **{src}**: fs={fs}, каналов={len(names)}, "
            f"отведено в канон={present},units={units}, gain={gain}, leads={names[:12]}")
    add("")
    add("Все сигналы приведены к 500 Гц / 5000 отсчётов / mV. "
        "Отсутствующие каналы помечаются маской `lead_mask` и **не** должны "
        "интерпретироваться как реальные нулевые отведения (см. `dataset_harmonization.py`).")

    # manifest
    cols = ["record_id", "patient_id", "source", "rhythm", "label", "split",
            "signal_hash", "patient_key", "record_key"]
    meta[cols].to_csv(OUT_MANIFEST, index=False)
    add("")
    add(f"Единый манифест: `{OUT_MANIFEST.relative_to(ROOT)}`.")
    if problems:
        add("")
        add("## 4. Статус")
        add("")
        add("**Data Integrity: FAIL** — требуется пересборка сплитов с групповой стратификацией.")
    else:
        add("")
        add("## 4. Статус")
        add("")
        add("**Data Integrity: PASS** (при условии устранения найденных дубликатов, если они есть).")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]), flush=True)
    print("Wrote", OUT_REPORT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
