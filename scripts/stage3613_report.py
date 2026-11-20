"""Stage 3.6.13 — cross-dataset generalisation report."""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3613"
OUT = ROOT / "reports" / "STAGE_3_6_13_CROSS_DATASET.md"


def main() -> int:
    df = pd.read_csv(MET / "cross_dataset_results.csv")
    lines = [
        "# Stage 3.6.13 — кросс-датасетная генерализация",
        "",
        "Направления: Chapman→PTB-XL, Chapman→CPSC-extra и обратно. "
        "Методы: без адаптации, CORAL, MMD, DANN. Адаптация использует только "
        "признаки целевого набора БЕЗ меток; порог выбран на source val и заморожен.",
        "",
        "| Источник | Цель | Метод | AUC source-val | AUC target | F1 target |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['source']} | {r['target']} | {r['method']} | "
                     f"{r['source_val_roc_auc']:.3f} | {r['target_roc_auc']:.3f} | "
                     f"{r['target_f1_macro']:.3f} |")
    lines += [
        "",
        "## Вывод",
        "",
        "- **Ни один метод доменной адаптации не обеспечил перенос**: целевые "
        "ROC-AUC 0.014–0.471 при source-val AUC 0.58–0.99.",
        "- CORAL/MMD дают небольшой прирост только на CPSC-extra (0.263→0.308), "
        "но не меняют качественную картину.",
        "- DANN нестабилен и не исправляет перекос: Chapman→PTB-XL AUC 0.023.",
        "- Обратный перенос PTB-XL→Chapman даёт AUC 0.446 (почти случайный), "
        "хотя внутри PTB-XL модель имеет AUC 0.96 — это прямое доказательство "
        "того, что модель учится source-specific признакам.",
        "",
        "**Итог:** domain generalization по-прежнему не доказана. Увеличение "
        "сложности модели без исправления разметки/источников не решает проблему.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
