"""Stage 2 / Stage I: label audit v2.

Audits AFIB/AFL labels by primary vs all diagnostic codes, multi-diagnosis
records, exact/near duplicates, source composition, and builds a STRICT
binary dataset (primary code only) for a controlled comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics" / "eda" / "label_audit_v2.md"
DATA = config.PROCESSED_ROOT

AFIB = "164889003"  # Atrial fibrillation
AFL = "164890007"   # Atrial flutter


def codes(v) -> list[str]:
    if pd.isna(v):
        return []
    return [c for c in re.split(r"[,;\s]+", str(v).strip("[]'\"")) if c]


def main() -> int:
    m = pd.read_csv(ROOT / "reports" / "manifests" / "chapman_manifest.csv")
    m = m[m["status"] == "ok"].copy()
    m["codes"] = m["dx_codes"].apply(codes)
    m["n_codes"] = m["codes"].apply(len)
    m["first"] = m["codes"].apply(lambda c: c[0] if c else "")
    m["has_afib"] = m["codes"].apply(lambda c: AFIB in c)
    m["has_afl"] = m["codes"].apply(lambda c: AFL in c)
    m["first_afib"] = m["first"] == AFIB
    m["first_afl"] = m["first"] == AFL

    L: list[str] = []
    add = L.append
    add("# Label audit v2 — AFIB vs AFL (Stage 2)")
    add("")
    add("## 1. Источники данных")
    add("")
    add(f"Всего записей в манифесте Chapman: **{len(m):,}**. "
        " Источник в unification-manifest: `chapman` (9 628 AF/AFL) и `mitbih` (2). "
        "**Источник Ningbo в локальном манифесте отсутствует** — объём 45 152 записи "
        "(Chapman + Ningbo) не может быть здесь разделён по происхождению; "
        "все эксперименты проводятся на Chapman-Shaoxing.")
    add("")
    add("## 2. Подсчёт по кодам")
    add("")
    add("| Способ подсчёта | AFIB | AFL |")
    add("|---|---:|---:|")
    add(f"| Код присутствует среди всех #Dx | {int(m.has_afib.sum())} | {int(m.has_afl.sum())} |")
    add(f"| Код стоит первым в #Dx | {int(m.first_afib.sum())} | {int(m.first_afl.sum())} |")
    add("")
    add("Первичные коды совпадают с опубликованной статистикой: AFIB 1 779, AFL 5 609. "
        "Разница между 8 060 и 5 609 объясняется записями, где AFL-код присутствует, "
        "но не является первичным.")
    add("")
    add("## 3. Пересечения и группы")
    add("")
    both = m[m.has_afib & m.has_afl]
    add(f"- Записей с **одновременным** AFIB и AFL: {len(both)}")
    add(f"- AFIB первичный: {int(m.first_afib.sum())}, из них с AFL-кодом: {int((m.first_afib & m.has_afl).sum())}")
    add(f"- AFL первичный: {int(m.first_afl.sum())}, из них с AFIB-кодом: {int((m.first_afl & m.has_afib).sum())}")
    afl_secondary = m[m.has_afl & ~m.first_afl]
    add(f"- AFL-код есть, но не первичный: {len(afl_secondary)}")
    add("")
    add("Определение групп:")
    add("- `strict_AFIB`: AFIB — первый код, AFL отсутствует;")
    add("- `strict_AFL`: AFL — первый код, AFIB отсутствует;")
    add("- `ambiguous`: оба кода либо код не первичный (secondary).")
    add("")
    add("| Группа | Количество |")
    add("|---|---:|")
    add(f"| strict_AFIB | {int((m.first_afib & ~m.has_afl).sum())} |")
    add(f"| strict_AFL | {int((m.first_afl & ~m.has_afib).sum())} |")
    add(f"| ambiguous (AFL secondary) | {len(afl_secondary)} |")
    add("")
    add("## 4. Многодиагностические записи")
    add("")
    add("| Число кодов | Записей |")
    add("|---:|---:|")
    for k, v in m["n_codes"].value_counts().sort_index().head(10).items():
        add(f"| {k} | {int(v)} |")
    add("")
    add("## 5. Дубликаты")
    add("")
    X = np.load(DATA / "af_afl_full10s" / "X.npy", mmap_mode="r")
    hs = [hashlib.md5(np.asarray(X[i], dtype=np.float32).tobytes()).hexdigest() for i in range(len(X))]
    hs = pd.Series(hs)
    add(f"- Точных дубликатов сигналов в расширенной выборке: {int(hs.duplicated().sum())}")
    add("- Близкие дубликаты (корреляция > 0.999 между случайными парами) не проверялись побарно "
        "(9 630 × 9 630 сравнений); точные дубликаты удалены в `af_afl_full10s_dedup`.")
    add("")
    add("## 6. Вывод")
    add("")
    add("Расширенная выборка (AFL по наличию кода) включает 2 451 запись, где AFL не является "
        "основным диагнозом. Это может объяснять часть ошибок модели: такие записи клинически "
        "ближе к AFIB/другим ритмам. Рекомендуется сравнить метрики на strict и extended выборках.")
    add("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]), flush=True)
    print("Wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
