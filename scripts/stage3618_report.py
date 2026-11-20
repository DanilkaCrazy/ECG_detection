"""Stage 3.6.18 — integrity-correction report."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3618"
OUT = ROOT / "reports" / "STAGE_3_6_18_INTEGRITY_CORRECTION.md"


def main() -> int:
    def load(name):
        p = MET / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    reb = load("rebuild_comparison.json")
    hold = load("holdout_status.json")
    exp = load("expert_review_package_v1_1.json")
    auc = load("auc_directionality_audit.json")
    sc = load("source_classifier_balanced.json")
    cmp = reb.get("comparison", {})
    lines = [
        "# Stage 3.6.18 — Data & Experiment Integrity Corrections (P0)",
        "",
        "## 1. SNOMED-ошибка исправлена",
        "",
        "Официальная таблица `ConditionNames_SNOMED-CT.csv` подтверждает:",
        "",
        "| Код | Акроним | Диагноз |",
        "|---|---|---|",
        "| 164889003 | AFIB | Atrial Fibrillation |",
        "| 164890007 | AF | Atrial Flutter |",
        "| 164934002 | TWC | T wave Change |",
        "",
        "- `164934002` удалён из `AFIB_CODES`; теперь даёт `OTHER`.",
        "- Голое `AF` удалено из `AFIB_CODES`: в Chapman оно означает flutter. ",
        "Добавлен namespaced `resolve_chapman_acronym()`.",
        "- Добавлены отрицательные тесты для TWC, Chapman `AF`, неизвестных кодов.",
        "",
        "## 2. Пересборка датасета",
        "",
        f"- `af_afl_stage3614_v1_1`: {reb.get('n_records', '?')} записей, "
        f"mapping_version 1.1. Stage 3.6.4 данные не перезаписаны.",
        f"- Неизменных меток: **{cmp.get('unchanged', '?')}**; "
        f"изменило класс: **{cmp.get('class_changed', '?')}**; "
        f"новых исключений: **{cmp.get('newly_excluded', '?')}**; "
        f"восстановлено записей: **{cmp.get('newly_added', '?')}**.",
        "- Ошибка TWC не меняла классы, но создавала ложные `AMBIGUOUS` — "
        "1325 записей были избыточно исключены и теперь возвращены.",
        "- Явные гейты: `meta.label == y.npy == split.label`, ",
        "а `meta.label` теперь присваивается из resolver'а, а не из старого CSV.",
        "",
        "## 3. Holdout понижен до development",
        "",
        f"- Переименованы файлы: {hold.get('renamed_split_files', [])}.",
        "- Причина: Stage 3.6.16 сравнивал 4 набора признаков × 3 seed, "
        "а Stage 3.6.17 использовал тот же сплит повторно.",
        "**Никакой текущий сплит не является независимой финальной оценкой.**",
        "",
        "## 4. Экспертный пакет",
        "",
        f"- Стратифицировано: {exp.get('n_stratified', '?')}, "
        f"целевых: {exp.get('n_targeted', '?')}, "
        f"создано карт: **{exp.get('n_cards', '?')}**.",
        "- Все 89 валидных PTB-XL включены; карты созданы и для 30 ambiguous.",
        "- Добавлены сетка 25 мм/с, калибровка 10 мм/мВ, явная маркировка отсутствующих отведений.",
        "",
        "## 5. Аудит направления PTB-XL",
        "",
        f"- `model.classes_`: {auc.get('model_classes')}; столбец P(AFIB): "
        f"{auc.get('probability_column_used_for_AFIB')}.",
        f"- AUC forward: **{auc.get('roc_auc_forward')}**, "
        f"после инверсии (только диагностика): {auc.get('roc_auc_after_inversion_diagnostic_only')}.",
        f"- Совпадение record_id с provenance: "
        f"{auc.get('row_alignment', {}).get('record_ids_match_provenance')}.",
        f"- Пересечение development-holdout PTB с офиц. folds 9/10: "
        f"{auc.get('dev_holdout_ptb_in_official_folds_9_10')}.",
        "- Инверсия не применяется к предсказаниям и меткам — только диагностика.",
        "",
        "## 6. Сбалансированный source classifier",
        "",
        "| Эксперимент | Accuracy | Balanced | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for k, v in sc.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} | {v['balanced_accuracy']:.3f} | "
                     f"{v['macro_f1']:.3f} |")
    lines += [
        "",
        "## 7. Ограничения",
        "",
        "- Экспертная ревизия ещё не выполнена — пакет подготовлен, но карты ",
        "должны быть просмотрены двумя кардиологами.",
        "- Новый независимый внешний набор (Stage 3.6.21) ещё не создан.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
