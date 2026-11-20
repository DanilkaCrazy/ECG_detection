"""Stage 3.6.22 — reproducibility and strict benchmark report."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3622"
OUT = ROOT / "reports" / "STAGE_3_6_22_REPRODUCIBILITY_BENCHMARK.md"


def main() -> int:
    b = json.loads((MET / "benchmark_summary.json").read_text(encoding="utf-8"))
    p = json.loads((ROOT / "reports" / "metrics" / "stage3621" /
                    "independent_protocol.json").read_text(encoding="utf-8"))
    es = json.loads((ROOT / "data" / "processed" / "af_afl_stage3622" /
                     "stats.json").read_text(encoding="utf-8"))
    lines = [
        "# Stage 3.6.22 — Reproducibility fixes & strict benchmark",
        "",
        "## 1. Что исправлено",
        "",
        "- `MAPPING_VERSION = 1.1` теперь единственный источник правды: `config.py`, ",
        "  `label_resolution.py`, `diagnosis_provenance.py`, `label_mapping.yaml`, ",
        "  `stage3614.yaml` и все манифесты датасетов.",
        "- Контракт независимой выборки усилен: проверка ≥2 источников, уникальных ",
        "  `signal_hash`, совпадения `label` и `expert_label` (или явной adjudication),",
        "  отсутствия пересечений с development- hashes.",
        "- Заморожены реальные артефакты E7: `frozen_E7.joblib`, preprocessing `.npz`, ",
        "  упорядоченный manifest признаков X+B+C (не все 372).",
        "- Признаки пересчитаны для **всех 7505** записей v1.1 — dependence ",
        "  от старой матрицы на 6180 записей устранена.",
        "- Карточки экспертов перегенерированы с физической калибровкой ",
        "  (1 mV = 10 мм, 0.2 с = 5 мм при 25 мм/с) и сеткой.",
        "- AV-модель: добавлена оценка надёжности, исправлена sample entropy ",
        "  (абсолютный порог 2% от среднего RR вместо 0.2·SD).",
        "",
        "## 2. Строгий benchmark",
        "",
        f"Записей: **{es['n_records']}**, признаков {es['n_features']}. ",
        f"Seed'ы: {b['seeds']}. Оценка — только development test.",
        "",
        "| Набор | Macro-F1 | AFIB-F1 | AFL-F1 | ROC-AUC | Balanced Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in b["mean_by_set"].items():
        lines.append(f"| {name} | {v['f1_macro']:.3f} | {v['afib_f1']:.3f} | "
                     f"{v['afl_f1']:.3f} | {v['roc_auc']:.3f} | "
                     f"{v['balanced_accuracy']:.3f} |")
    lines += ["", "## 3. Bootstrap (учитывает и строки, и seed)", ""]
    for name, v in b["bootstrap_vs_E0"].items():
        lines.append(f"- **{name} vs E0**: {v['mean']:+.4f}, 95% CI "
                     f"[{v['lo']:+.4f}; {v['hi']:+.4f}], значимо = **{v['significant']}**.")
    mc = MET / "model_comparison_summary.json"
    if mc.exists():
        m = json.loads(mc.read_text(encoding="utf-8"))
        lines += ["", "## 3b. Сравнение моделей (E2/E7)", "",
                  "| Набор | Модель | F1 | AFIB-F1 | ROC-AUC |", "|---|---|---:|---:|---:|"]
        for key, v in m["mean_by_model_set"].items():
            s, name = key.split("|")
            lines.append(f"| {s} | {name} | {v['f1_macro']:.3f} | {v['afib_f1']:.3f} | "
                         f"{v['roc_auc']:.3f} |")
        for name, v in m["bootstrap_vs_RF_on_E2"].items():
            lines.append(f"- **{name} − RF на E2**: {v['mean']:+.4f}, 95% CI "
                         f"[{v['lo']:+.4f}; {v['hi']:+.4f}], значимо = {v['significant']}.")
    lines += [
        "",
        "## 4. Статус независимой оценки",
        "",
        f"**{p['status']}**. Причины: " + "; ".join(p["violations"]) + ".",
        f"SHA256 замороженной модели: `{p['frozen_artifacts'].get('model_sha256')}`.",
        f"Признаков в замороженном manifest: {p['frozen_artifacts'].get('n_features')}.",
        "",
        "## 5. Вывод",
        "",
        "E2 остаётся более консервативной моделью для AFIB-F1; E7 — кандидат ",
        "по общему Macro-F1. H и E7_AV обсуждаются только при статистическом ",
        "подтверждении. Ни один результат не является независимой внешней оценкой.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
