"""Stage 3.6.15 — source leakage & harmonization report."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3615"
OUT = ROOT / "reports" / "STAGE_3_6_15_SOURCE_LEAKAGE.md"


def main() -> int:
    sig = json.loads((MET / "signal_audit.json").read_text(encoding="utf-8"))
    sc = json.loads((MET / "source_classifier.json").read_text(encoding="utf-8"))
    dom = pd.read_csv(MET / "domain_evaluation_results.csv")
    harm = pd.read_csv(MET / "harmonization_variants.csv")

    lines = [
        "# Stage 3.6.15 — Source Leakage & Signal Harmonization (P0)",
        "",
        "Дата: 2026-11-20. Аудит performs on `af_afl_stage3614` (6180 записей с ",
        "проверенным происхождением меток).",
        "",
        "## 1. Физические характеристики источников",
        "",
        "| Источник | n | RMS | p99 | zero-frac |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sig["by_source"]:
        lines.append(f"| {r['source']} | {int(r['n'])} | {r['rms_mean']:.3f} | "
                     f"{r['p99_mean']:.3f} | {r['zero_frac_mean']:.4f} |")
    lines += [
        "",
        "PTB-XL имеет заметно меньшую амплитуду (RMS 0.225 против 0.278 у Chapman). "
        "Это может отражать аппаратуру/фильтры, но не доказывает физиологическое различие.",
        "",
        "## 2. Source classifier",
        "",
        "| Эксперимент | Accuracy | Balanced Acc | ROC-AUC OVR |",
        "|---|---:|---:|---:|",
    ]
    for name in ("A_metadata", "B_raw_ecg"):
        r = sc[name]
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | "
                     f"{r['roc_auc_ovr']:.3f} |")
    for name, r in sc["C_class_conditional"].items():
        lines.append(f"| C_{name} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | — |")
    lines += [
        "",
        "Источник распознаётся по сигналу даже при ФИКСИРОВАННОМ классе: "
        "AFIB 0.979, AFL 0.989 accuracy. Значит различие источников не сводится "
        "к разной доле классов. Признаки построены только из сигнала — ID, пути "
        "и метки в них не попадают.",
        "",
        "## 3. Исправленный UDA-протокол",
        "",
        "Целевой домен разделён на adaptation (50%, без меток) и locked test (50%). "
        "Три режима Reported отдельно:",
        "",
        "| Режим | Направление | Метод | Целевой AUC | Целевой F1 |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in dom.iterrows():
        lines.append(f"| {r['regime']} | {r['source']}→{r['target']} | {r['method']} | "
                     f"{r['roc_auc']:.3f} | {r['f1_macro']:.3f} |")
    lines += [
        "",
        "Inductive UDA на независимом целевом тесте не превысила 0.368 AUC "
        "(CPSC, DANN) и 0.294 (PTB-XL, DANN). transductive-режим формально "
        "выше, но eval'ируется на тех же записях, что и адаптация — "
        "это не доказательство обобщения.",
        "",
        "## 4. Варианты предобработки (H0–H4)",
        "",
        "| Вариант | F1 | AFIB-F1 | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in harm.iterrows():
        lines.append(f"| {r['variant']} | {r['f1_macro']:.3f} | {r['afib_f1']:.3f} | "
                     f"{r['roc_auc']:.3f} | {r['brier']:.3f} |")
    lines += [
        "",
        "## 5. Исправления preprocessing",
        "",
        "- `stage3611_prepare.py`: простая децимация `sig[:, ::4]` заменена на "
        "`decimate_signal` (`resample_poly` с антиалиасинговой фильтрацией).",
        "- `stage3611_train.py`, `stage3612_*.py`: нормализация по оси 1 "
        "(межотведенческая) заменена на нормализацию каждого отведения по времени.",
        "- `stage3612_transfer.py`: preprocessing эмбеддингов теперь обучается "
        "только на train и применяется одинаково к val/test/external.",
        "- Старые результаты сохранены (`modular_results.json`, `transfer_results.json`); "
        "новые — с суффиксом `_corrected`.",
        "",
        "## 6. Ограничения",
        "",
        "- H5 (сравнение 125/250/500 Гц) не завершено — требуется повторная "
        "экстракция признаков на трёх частотах.",
        "- Алиас `AF → aVF` в `dataset_harmonization.py` остаётся "
        "непроверенным по документации конкретного источника; помечено как допущение.",
        "- Экспертная ревизия меток ещё не выполнена (подготовлен только пакет).",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
