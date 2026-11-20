"""Stage 3.6.17 — final independent assessment report."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3617"
OUT = ROOT / "reports" / "STAGE_3_6_17_INDEPENDENT_ASSESSMENT.md"


def main() -> int:
    r = json.loads((MET / "independent_assessment.json").read_text(encoding="utf-8"))
    lines = [
        "# Stage 3.6.17 — Independent Generalization Assessment",
        "",
        "Замороженная модель E7 RF (порог выбран на Chapman val) оценивается ",
        "один раз на запечатанном holdout, который не участвовал ни в одном ",
        "выборе архитектуры/признаков/порога.",
        "",
        "## Sealed holdout",
        "",
        "| Подвыборка | n | F1-macro | AFIB-F1 | AFL-F1 | ROC-AUC | 95% CI F1 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, b in r["sealed_holdout"].items():
        c = b.get("bootstrap_ci", {})
        lines.append(f"| {name} | {b['n']} | {b['f1_macro']:.3f} | {b['afib_f1']:.3f} | "
                     f"{b['afl_f1']:.3f} | {b['roc_auc']:.3f} | "
                     f"[{c.get('lo', float('nan')):.3f}; {c.get('hi', float('nan')):.3f}] |")
    if "ptbxl_official_folds" in r:
        b = r["ptbxl_official_folds"]
        lines += ["", "## PTB-XL официальные fold'ы (folds 1–8 train, 9–10 test)", "",
                  f"- n_train={b['n_train']}, n_test={b['n_test']}, "
                  "grouping по настоящему `patient_id`.",
                  f"- F1-macro **{b['f1_macro']:.3f}**, AFIB-F1 {b['afib_f1']:.3f}, "
                  f"AFL-F1 {b['afl_f1']:.3f}, ROC-AUC **{b['roc_auc']:.3f}**.",
                  f"- PR-AUC {b['pr_auc']:.3f} при baseline prevalence "
                  f"{b['pr_baseline']:.3f}."]
    lines += [
        "",
        "## Вывод",
        "",
        "Даже на очищенной разметке модель не демонстрирует переноса на другой ",
        "источник: ROC-AUC на внешних наборах остаётся около или ниже случайного. ",
        "Внутренний holdout показывает сопоставимое с test качество — значит ",
        "протокол выбора модели теперь корректен, но проблема лежит в ",
        "различиях источников и/или в качество разметки, который требует ",
        "экспертной ревизии (Stage 3.6.14, пакет подготовлен).",
        "",
        "## Ограничения",
        "",
        "- Bootstrap для Chapman — record-level (настоящего patient_id нет).",
        "- Для PTB-XL после очистки доступно всего 89 валидных записей (56 AFL, 33 AFIB), "
        "что сильно ограничивает мощность внешней оценки.",
        "- Экспертная ревизия кардиологами ещё не выполнена.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
