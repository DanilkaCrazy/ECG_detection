"""Stage 3.6.20/3.6.21 — AV model and independent protocol reports."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT20 = ROOT / "reports" / "STAGE_3_6_20_AV_CONDUCTION_MODEL.md"
OUT21 = ROOT / "reports" / "STAGE_3_6_21_INDEPENDENT_PROTOCOL.md"


def main() -> int:
    r = json.loads((ROOT / "reports" / "metrics" / "stage3620" /
                    "av_ablation.json").read_text(encoding="utf-8"))
    p = json.loads((ROOT / "reports" / "metrics" / "stage3621" /
                    "independent_protocol.json").read_text(encoding="utf-8"))
    lines = [
        "# Stage 3.6.20 — Physiological AV-conduction features",
        "",
        "Модуль `av_conduction_model.py` оценивает соответствие RR-последовательности ",
        "моделям предсердного цикла с AV-блокадой 2:1/3:1/4:1. Признаки: CV, RMSSD, ",
        "SDNN, Poincaré SD1/SD2, sample entropy, автокорреляционный лаг, оценённый ",
        "предсердный цикл, согласованность и ошибка модели, число уровней блокады.",
        "",
        "| Набор признаков | F1-macro | AFIB-F1 | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for k, v in r["results"].items():
        lines.append(f"| {k} | {v['f1_macro']:.3f} | {v['afib_f1']:.3f} | "
                     f"{v['roc_auc']:.3f} |")
    lines += ["", "## Статистическая проверка (vs E7)", ""]
    for k, b in r["bootstrap_vs_E7"].items():
        lines.append(f"- **{k} − E7**: {b['mean']:+.4f}, 95% CI "
                     f"[{b['lo']:+.4f}; {b['hi']:+.4f}], значимо = **{b['significant']}**.")
    lines += [
        "",
        "**Вывод:** AV-признаки сами по себе дают сопоставимый результат (F1 0.542), ",
        "но не дают статистически значимого прироста при добавлении к E7 ",
        "(ΔF1 +0.003, CI включает нуль). Оценка проведена только на ",
        "train/val/development-test; пониженный holdout не использовался.",
        "",
    ]
    OUT20.write_text("\n".join(lines), encoding="utf-8")

    lines21 = [
        "# Stage 3.6.21 — Independent evaluation protocol",
        "",
        f"**Статус: {p['status']}**.",
        "",
        "Причины блокировки:",
        "",
    ]
    for v in p["violations"]:
        lines21.append(f"- {v}.")
    lines21 += [
        "",
        "Требования: ≥100 записей каждого класса с экспертными метками, ",
        "настоящие patient_id, ≥2 источников, отсутствие пересечений с development.",
        "",
        "## Заморозка перед единственной финальной оценкой",
        "",
        json.dumps(p["frozen_artifacts"], indent=2, ensure_ascii=False),
        "",
        "После заморозки запрещено подстраивать порог, признаки и preprocessing ",
        "по меткам нового набора.",
        "",
    ]
    OUT21.write_text("\n".join(lines21), encoding="utf-8")
    print("Wrote", OUT20, OUT21)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
