"""Stage 3.6.14 — final data provenance report."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3614"
OUT = ROOT / "reports" / "STAGE_3_6_14_DATA_PROVENANCE.md"


def main() -> int:
    raw = json.loads((MET / "raw_diagnosis_audit.json").read_text(encoding="utf-8"))
    aud = json.loads((MET / "label_audit.json").read_text(encoding="utf-8"))
    amb = json.loads((MET / "ambiguity_analysis.json").read_text(encoding="utf-8"))
    ds = json.loads((ROOT / "data" / "processed" / "af_afl_stage3614" /
                     "stats.json").read_text(encoding="utf-8"))
    by = pd.DataFrame(aud["by_source"])
    lines = [
        "# Stage 3.6.14 — Data Provenance & Label Resolution (P0)",
        "",
        "Все исходные диагностические коды перечитаны из первоисточников: Chapman/CPSC ",
        "— из WFDB `#Dx`, PTB-XL — из `scp_codes` + `scp_statements.csv`. ",
        "Старый `meta.rhythm`/`meta.label` при проверке НЕ использовался.",
        "",
        "## 1. Каноническое соответствие",
        "",
        "| Код | Диагноз | Класс | Label |",
        "|---|---|---|---:|",
        "| SNOMED 164889003 | Atrial fibrillation | AFIB | 1 |",
        "| SNOMED 164890007 | Atrial flutter | AFL | 0 |",
        "| SCP AFIB | Atrial fibrillation | AFIB | 1 |",
        "| SCP AFLT | Atrial flutter | AFL | 0 |",
        "",
        "Правило: запись с ОБОИМИ кодами получает статус `AMBIGUOUS` и бинарной метки ",
        "не получает. Код с likelihood = 0 считается ОТСУТСТВУЮЩИМ утверждением.",
        "",
        "## 2. Исправленные ошибки прежнего pipeline",
        "",
        "1. `config.CHAPMAN_SNOMED_TO_ACRONYM`: 164889003 → `AF`, 164890007 → `AFL` ",
        "(ранее 164889003 → `AFIB`, 164890007 → `AF` — противоречие).",
        "2. `map_ptbxl_scp_to_rhythm`: при AFIB+AFLT теперь `None`, а не AF.",
        "3. `map_chapman_dx_to_rhythm`: при AFIB+AFL теперь `None`, а не первый код.",
        "4. `build_external_af_dataset` проверял только наличие ключа `AFIB` ",
        "без учёта likelihood — исправлено в центральном resolver'е.",
        "",
        "## 3. Результат аудита 8962 записей",
        "",
        "| Источник | Всего | Валидно | Ambiguous | Other | Невалидных старых меток |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in by.iterrows():
        lines.append(f"| {r['source']} | {int(r['n'])} | {int(r['n_valid'])} | "
                     f"{int(r['n_ambiguous'])} | {int(r['n_other'])} | "
                     f"{int(r['n_old_label_invalid'])} |")
    lines += [
        "",
        f"- Всего валидных: **{aud['n_valid']}** из {aud['n_records']}.",
        f"- Старых меток без валидного исходного диагноза: **"
        f"{aud['n_old_labels_without_valid_diagnosis']}**.",
        f"- Записей сlikelihood 0 по целевому коду: **"
        f"{amb['total_zero_likelihood_target']}** (практически все — PTB-XL AFIB).",
        "- Среди валидных записей расхождений со старыми метками **нет** — "
        "проблема была не в перепутанных классах, а в включении записей, "
        "у которых диагноз в источнике отсутствует.",
        "",
        "**Ключевой вывод:** низкие внешние ROC-AUC ранее объяснялись не только "
        "domain shift, но и массовой ошибкой разметки PTB-XL: 1457 из 1546 "
        "записей имели AFIB с likelihood 0.0 и были включены как AFIB. "
        "Domain shift вероятен, но его размер нужно переоценить на чистой выборке.",
        "",
        "## 4. Новая версия датасета",
        "",
        f"- `data/processed/af_afl_stage3614/`: **{ds['n_records']}** записей.",
        f"- По источникам: {ds['by_source']}; по классам: {ds['by_label']}.",
        f"- Split: {ds['by_split']}; нарушений изоляции: **{len(ds['split_violations'])}**.",
        "- Stage 3.6.4 данные не перезаписаны; manifest содержит SHA256 всех файлов.",
        "",
        "## 5. Ограничения",
        "",
        "- Экспертная проверка кардиологами ещё не проведена — подготовка пакета ",
        "описана отдельно. Машинная валидность меток ≠ клиническая подтверждённость.",
        "- Для Chapman `patient_key` — суррогат записи; patient-level анализ невозможен.",
        "- 1324 ambiguous-записи Chapman исключены консервативно; это может сместить ",
        "распределение классов и требует отдельного анализа.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
