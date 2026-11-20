"""Stage 3.6.11 — modular architecture report."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3611"
OUT = ROOT / "reports" / "STAGE_3_6_11_MODULAR_ARCHITECTURE.md"


def main() -> int:
    r = json.loads((MET / "modular_results.json").read_text(encoding="utf-8"))
    lines = [
        "# Stage 3.6.11 — модульная архитектура AFIB/AFL",
        "",
        "Три независимых энкодера: RR/ритм, предсердный остаток, морфология сырого сигнала.",
        "Attention Fusion даёт веса модулей как артефакт интерпретируемости. "
        "**Внимание:** энкодер предсердного остатка потребляет уже подготовленный "
        "детерминированный остаток — сеть НЕ обучается разделять QRST и F-волны "
        "end-to-end. Веса модулей показывают, какой поток данных использован, "
        "а не физиологическую декомпозицию.",
        "",
        "Протокол: fit=Chapman train, порог=Chapman val, контроль=Chapman test, "
        "внешние наборы — с замороженным порогом. Сигналы децимированы 5000→1250 "
        "(эффективные 125 Гц) для CPU-обучения.",
        "",
        f"Порог: **{r['threshold']:.2f}**, эпох: {r['epochs']}, время: {r['total_seconds']:.1f} с.",
        "",
        "| Выборка | F1-macro | AFIB-F1 | AFL-F1 | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("chapman_test", "ptbxl", "cpsc2018_extra"):
        b = r[name]
        lines.append(f"| {name} | {b['f1_macro']:.3f} | {b['afib_f1']:.3f} | "
                     f"{b['afl_f1']:.3f} | {b['roc_auc']:.3f} | {b['brier']:.3f} |")
    w = r["mean_module_weights"]
    lines += [
        "",
        "## Интерпретируемость: веса модулей",
        "",
        f"- RR/ритм: **{w['rhythm']:.3f}**",
        f"- Предсердный остаток: **{w['atrial']:.3f}**",
        f"- Морфология: **{w['morphology']:.3f}**",
        "",
        "**Критический результат:** fuse-механизм почти полностью опирается на "
        "сырую морфологию (~77%), а не на физиологически интерпретируемые потоки. "
        "Это объясняет, почему F1 (0.517) ниже признакового E7 RF (0.571): "
        "модульная сеть пока не использует предсердную информацию в полной мере. "
        "Возможные причины: децимация до 125 Гц, малый объём train (2433 записи), "
        "отсутствие auxiliary-потери на предсердный остаток.",
        "",
        "Внешние ROC-AUC: PTB-XL **0.403**, CPSC **0.449** — выше, чем у "
        "признакового RF (0.069/0.35), но всё ещё ниже 0.5. Domain generalization "
        "не доказана.",
        "",
        "## Следующие шаги",
        "",
        "- auxiliary reconstruction loss на остатке, чтобы заставить atrial-ветвь учиться;",
        "- балансировка весов модулей (entropy/дроп-аут), чтобы избежать коллапса в морфологию;",
        "- обучение на полной частоте 500 Гц при наличии GPU.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
