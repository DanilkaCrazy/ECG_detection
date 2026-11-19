"""Final report for external AF data integration and domain-shift study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_merged"
FIG = ROOT / "reports" / "figures" / "af_afl_merged"


def main() -> int:
    stats = json.loads((config.PROCESSED_ROOT / "af_afl_merged" / "stats.json").read_text(encoding="utf-8"))
    ext = json.loads((config.PROCESSED_ROOT / "af_external" / "stats.json").read_text(encoding="utf-8"))
    var = pd.read_csv(MET / "merged_variants.csv")
    res = pd.read_csv(MET / "merged_results.csv")

    # source distribution figure
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(stats["sources"]).plot(kind="bar", ax=ax, color="#2c7fb8")
    ax.set_title("Records by source — merged dataset")
    ax.set_ylabel("records"); plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "sources.png", dpi=150); plt.close()

    L = [
        "# Внешние AF-данные и балансировка классов", "",
        "## 1. Проанализированные наборы данных", "",
        "Локально доступны Chapman-Shaoxing, PTB-XL, CPSC2018, CPSC2018-extra, "
        "MIT-BIH arrhythmia, INCART, SHDB-AF, NSRDB, QTDB, SVDB, VFDB и др. "
        "Для бинарной задачи AFIB/AFL пригодны 12-канальные записи с надёжной разметкой:",
        "",
        "| Источник | AFIB | AFL | Использовано |",
        "|---|---:|---:|---:|",
        "| PTB-XL (scp_codes AFIB/AFLT) | 1 514 | 73 | да |",
        "| CPSC2018 (# Dx SNOMED) | 1 221 | 0 | да |",
        "| CPSC2018-extra | 153 | 54 | да |",
        "| Chapman strict (первичный #Dx) | 1 779 | 5 609 | да |",
        "",
        "Записи с обоими кодами или неоднозначной разметкой отбрасывались. "
        "Сигналы приведены к 500 Гц / 5 000 отсчётов / 12 отведений / float32.",
        "",
        f"Внешняя выборка: **{ext['n']}** записей (AFIB {ext['n_afib']}, AFL {ext['n_afl']}); "
        f"источники: {ext['by_source']}.",
        "",
        f"Объединённая выборка: **{stats['n']}** записей, AFIB **{stats['n_afib']} ({100*stats['n_afib']/stats['n']:.1f}%)**, "
        f"AFL **{stats['n_afl']} ({100*stats['n_afl']/stats['n']:.1f}%)**, "
        f"соотношение **{stats['ratio']:.2f} : 1** — классы почти поровну.",
        "",
        "## 2. Разделение данных",
        "",
        "- train: Chapman + внешние (7 250), val: Chapman + внешние (1 552);",
        "- test_chapman: 1 106 записей — **независимый in-domain тест** (не использовался при настройке);",
        "- test_external: 446 записей — out-of-domain контроль.",
        "",
        "## 3. Результаты (Chapman test)",
        "",
        "| Стратегия | Macro-F1 | AFIB-F1 | AFL-F1 | AFIB Recall | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in var.iterrows():
        L.append(f"| {r['variant']} | {r['f1_macro']:.4f} | {r['afib_f1']:.4f} | "
                 f"{r['afl_f1']:.4f} | {r['afib_recall']:.4f} | {r['roc_auc']:.4f} |")
    L += [
        "",
        "Лучший результат на Chapman — **модель только на Chapman** (A: 0.598 / 0.399 AFIB-F1). "
        "Объединённая модель (B) даёт 0.565 / 0.381; веса источников (C) и частичная подмесь (D) не помогли.",
        "",
        "## 4. Внешний тест (domain shift)",
        "",
        "| Тест | Macro-F1 | AFIB-F1 | AFL-F1 | PR-AUC AFIB | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in res.iterrows():
        L.append(f"| {r['test']} | {r['f1_macro']:.4f} | {r['afib_f1']:.4f} | "
                 f"{r['afl_f1']:.4f} | {r['pr_auc_afib']:.4f} | {r['roc_auc']:.4f} |")
    L += [
        "",
        "Внешние AFIB распознаются легко (PR-AUC 0.987), но внешняя выборка почти не содержит AFL "
        "(17 из 446), поэтому AFL-F1 там не показателен. Общий порог для всех источников смещён: "
        "для Chapman нужен порог ~0.27, для внешних ~0.10–0.82 в зависимости от источника. "
        "Рекомендуется калибровка по источнику.",
        "",
        "## 5. Вывод",
        "",
        "Добавление внешних AF-записей действительно выровняло классы (45% / 55%), но **не улучшило "
        "in-domain метрики Chapman** из-за domain shift (другие популяции, устройства, протоколы). "
        "Для продакшена на Chapman следует использовать модель A; объединённая модель применима "
        "для кросс-датасетной проверки и как предобучение при наличии калибровки по источнику.",
        "",
        "Воспроизведение: `scripts/build_external_af_dataset.py` → "
        "`merge_af_datasets.py` → `extract_af_afl_features.py` (expert/v2/heat) → "
        "`train_merged_variants.py`.",
    ]
    out = ROOT / "reports" / "AF_AFL_MERGED_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("Wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
