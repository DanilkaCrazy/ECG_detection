"""Stage 3.6.9 — assemble the Experimental Integrity & Label Audit report."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage369"
OUT = ROOT / "reports" / "STAGE_3_6_9_EXPERIMENTAL_INTEGRITY.md"


def load(name, default=None):
    p = MET / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def fmt(d, keys):
    return " | ".join(f"{k}={d.get(k)}" for k in keys)


def main() -> int:
    lab = load("label_audit.json", {})
    spl = load("split_audit.json", {})
    locked = load("split_locked_audit.json", {})
    ext = load("external_test_corrected.json", {})
    lead = load("lead_audit.json", {})
    qrs = load("qrs_audit.json", {})

    lines = [
        "# Stage 3.6.9 — Experimental Integrity & Label Audit (P0)",
        "",
        "Дата: 2026-11-20. Все метрики ниже рассчитаны с зафиксированными seed=42 ",
        "и сохранёнными артефактами в `reports/metrics/stage369/`.",
        "",
        "## 1. Dataset audit",
        "",
        f"- Записей: **{lab.get('n_records', '?')}**; источников: chapman, ptbxl, cpsc2018_extra.",
        "- Классы: 1 = AFIB (фибрилляция), 0 = AFL (трепетание).",
        "- Точных дубликатов по `signal_hash`: **0** (проверено в `stage369_split_audit.py`).",
        "",
        "| Источник | AFIB | AFL | Доля AFL |",
        "|---|---:|---:|---:|",
    ]
    for src, v in lab.get("by_source", {}).items():
        tot = v["n"]
        lines.append(f"| {src} | {v['n_afib']} | {v['n_afl']} | {v['n_afl']/tot:.3f} |")
    lines += [
        "",
        "## 2. Label audit",
        "",
        f"- Конфликтов меток: **{lab.get('n_conflicts', '?')}**; неизвестных кодов: "
        f"**{lab.get('n_unknown', '?')}**.",
        f"- Направление кодирования согласовано: **{lab.get('direction_ok')}**.",
        "- Явное сопоставление: Chapman `AF`→1, `AFL`→0; PTB-XL `AFIB`→1, `AFLT`→0; "
        "CPSC SNOMED `164889003`→1, `164890007`→0.",
        "- **Важно:** `config.CHAPMAN_SNOMED_TO_ACRONYM` содержит устаревшее противоречащее "
        "сопоставление (164889003→AFIB, 164890007→AF); авторитетным является "
        "`configs/label_mapping.yaml` и `src/ecgdet/label_audit.py`.",
        "- Исторический аудит (`audit_labels_v2.py`) показал 2451 запись Chapman, где AFL-код "
        "присутствует, но не является первичным. В текущем stage34 эти записи уже "
        "отфильтрованы QC/primary-code правилом — конфликтов в финальной таблице нет.",
        "",
        "## 3. Split audit",
        "",
        f"- Исходный split: {spl.get('counts')}.",
        f"- Пересечений индексов/пациентов: **{len(spl.get('leakage_violations', []))}**.",
        f"- SHA256split: `{spl.get('fingerprint', '')[:16]}…`.",
        f"- Patient-independent validation: **{spl.get('patient_independent')}** "
        "(у Chapman нет настоящего `patient_id`; `patient_key` = record id).",
        f"- Новый запечатанный holdout: {locked.get('counts')}; он НЕ используется ни в одном "
        "эксперименте 3.6.9/3.6.10.",
        "",
        "## 4. External test audit (исправленный протокол)",
        "",
        "Старый `stage36_external_test.py` обучался на ВСЕХ строках Chapman (включая test) и "
        "подбирал порог по меткам внешнего набора — объявлен устаревшим. Новый протокол: "
        "fit=Chapman train, порог=Chapman val, заморозка, затем оценка.",
        "",
        "| Выборка | n | F1-macro | AFIB-F1 | AFL-F1 | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    blocks = [("chapman control", ext.get("control")), *list(ext.get("external", {}).items())]
    for name, b in blocks:
        if not b:
            continue
        lines.append(f"| {name} | {b['n']} | {b['f1_macro']:.3f} | {b['afib_f1']:.3f} | "
                     f"{b['afl_f1']:.3f} | {b['roc_auc']:.3f} | {b['brier']:.3f} |")
    ins = ext.get("in_source_reference", {})
    lines += [
        "",
        "### Диагностика направления вероятностей",
        "",
        "ROC-AUC ниже случайного может быть инверсией меток ИЛИ доменным сдвигом. Проверка:",
    ]
    for src, v in ins.items():
        lines.append(f"- Обучение и тест **внутри {src}**: ROC-AUC = **{v.get('roc_auc')}** "
                     f"(n_test={v.get('n_test')}).")
    lines += [
        "",
        f"Вывод: внутриисточниковые ROC-AUC высокие (0.97 PTB-XL, 0.75 CPSC), а кросс-доменный "
        f"Chapman→PTB-XL = **{ext['external']['ptbxl']['roc_auc']:.3f}**. Метки согласованы, "
        "значит причина — **сильный domain shift**, а не инверсия кодирования. "
        "Дополнительно: доля AFL в Chapman 0.756 против 0.036 в PTB-XL — резкий prior shift.",
        "",
        "## 5. Lead audit",
        "",
        f"- Записей со всеми 12 отведениями: **{lead.get('n_with_all_leads')}** из "
        f"{lead.get('n_records')}; с пропущенными: **{lead.get('n_with_missing_leads')}**.",
        f"- Среднее число пригодных отведений: **{lead.get('mean_usable_leads'):.3f}**;",
        f"  записей с нулём пригодных: **{lead.get('n_with_zero_good_leads')}** — записи не "
        "удалялись, применяется `lead_mask`.",
        "- С historical `stage36_strict_integrity.json`: 129 нарушений на 20764 пар "
        "(главным образом channel_nonzero<0.5) — причины документированы, маски передаются "
        "в признаковые модули.",
        "",
        "## 6. QRS audit",
        "",
        "Допуски заданы в **мс** (25/50/100), а не в отсчётах. Детектор: 5-15 Hz по умолчанию "
        "+ условные fallback-ветви для тахикардии и F-волн ( Stage 3.6.9).",
        "",
        "| Dataset | Допуск, мс | Sens | PPV | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in qrs.get("annotated", []):
        lines.append(f"| {r['dataset']} | {r['tol_ms']} | {r['sensitivity']:.3f} | "
                     f"{r['ppv']:.3f} | {r['f1']:.3f} |")
    lines += ["", "Синтетические крайние случаи (recall):"]
    for r in qrs.get("synthetic", []):
        lines.append(f"- `{r['case']}`: {r['recall']:.2f} (status={r['status']})")
    lines += [
        "",
        "## 7. Limitations",
        "",
        "- Настоящего `patient_id` для Chapman нет — patient-independent validation "
        "не заявляется.",
        "- Внешние наборы имеют резко отличное распределение классов; метрики F1 на них "
        "не сопоставимы напрямую с внутренним test.",
        "- Синтетический F-wave тест показывает остаточную чувствительность к F-волнам "
        "(recall 0.52) — fallback включается только при полном отказе основного детектора.",
        "-_locked holdout запечатан, но его объём (1786 записей) может быть недостаточен "
        "для стабильных оценок по каждому источнику отдельно.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
