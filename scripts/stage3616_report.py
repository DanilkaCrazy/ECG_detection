"""Stage 3.6.16 — corrected benchmark report."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3616"
OUT = ROOT / "reports" / "STAGE_3_6_16_CORRECTED_BENCHMARK.md"


def main() -> int:
    df = pd.read_csv(MET / "benchmark_results.csv")
    s = json.loads((MET / "benchmark_summary.json").read_text(encoding="utf-8"))
    agg = df.groupby("experiment").mean(numeric_only=True)
    lines = [
        "# Stage 3.6.16 — Corrected Model Benchmark",
        "",
        "Датасет `af_afl_stage3614` (6180 записей, метки восстановлены из исходных диагнозов). ",
        "Признаки проверены по `signal_hash` (0 расхождений) и подмножеством перенесены ",
        "из stage34/stage36/stage3610. Протокол: fit=Chapman train, порог=Chapman val, ",
        "3 seed; test — контроль, sealed holdout — только после заморозки.",
        "",
        "| Эксперимент | Chapman F1 | Holdout F1 | Chapman AUC | PTB-XL AUC | CPSC AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for eid, r in agg.iterrows():
        lines.append(f"| {eid} | {r['chapman_test_f1_macro']:.3f} | "
                     f"{r['sealed_holdout_f1_macro']:.3f} | "
                     f"{r['chapman_test_roc_auc']:.3f} | {r['ptbxl_roc_auc']:.3f} | "
                     f"{r['cpsc2018_extra_roc_auc']:.3f} |")
    lines += ["", "## Статистическая проверка (парный bootstrap vs E0)", ""]
    for eid, b in s["bootstrap_vs_E0"].items():
        lines.append(f"- **{eid} − E0**: {b['mean']:+.4f}, 95% CI "
                     f"[{b['lo']:+.4f}; {b['hi']:+.4f}], значимо = **{b['significant']}**.")
    lines += [
        "",
        "**Ключевой вывод:** на очищенной выборке прежний выигрыш E7 перестал быть "
        "статистически значимым. E2/E7 дают F1 0.558/0.563 против 0.538 у E0, "
        "но CI включают нуль. На запечатанном holdout E7 (0.516) даже ниже E2 (0.542). "
        "Внешние ROC-AUC остаются ниже случайного уровня (PTB-XL 0.003–0.027 на 89 "
        "валидных записях). Прежний прирост +0.041 на stage34 частично обусловлен "
        "2782 невалидными метками.",
        "",
        "## Сравнение частот дискретизации (H5)",
        "",
    ]
    rp = MET / "resampling_comparison.csv"
    if rp.exists():
        rr = pd.read_csv(rp)
        lines += ["| Частота | F1 | AFIB-F1 | ROC-AUC |", "|---|---:|---:|---:|"]
        for _, r in rr.iterrows():
            lines.append(f"| {int(r['fs'])} Гц | {r['f1_macro']:.3f} | "
                         f"{r['afib_f1']:.3f} | {r['roc_auc']:.3f} |")
    lines += ["", "## Ограничения", "",
              "- PTB-XL после очистки содержит только 89 валидных записей — "
              "внешние метки крайне нестабильны.",
              "- Bootstrap для Chapman — record-level (настоящего patient_id нет).",
              "- Аудит экспертами ещё не проведён; машинная валидность ≠ клиническая.",
              ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
