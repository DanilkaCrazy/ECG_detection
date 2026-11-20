"""Stage 3.6.10 — ablation and feature-importance report generator."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3610"
OUT1 = ROOT / "reports" / "STAGE_3_6_10_FEATURE_ABLATION.md"
OUT2 = ROOT / "reports" / "STAGE_3_6_10_FEATURE_IMPORTANCE.md"


def main() -> int:
    df = pd.read_csv(MET / "ablation_results.csv")
    boot = json.loads((MET / "ablation_bootstrap.json").read_text(encoding="utf-8"))
    best = json.loads((MET / "ablation_summary.json").read_text(encoding="utf-8"))["best"]

    lines = [
        "# Stage 3.6.10 — Feature group ablation (E0–E8)",
        "",
        "Протокол одинаков для всех экспериментов: fit = Chapman train, порог = Chapman val, "
        "контроль = Chapman test, внешние наборы оцениваются с ЗАМОРОЖЕННЫМ порогом. "
        "Бутстреп — парный по `patient_key`, 1000 ресемплов, коррекция Benjamini–Hochberg.",
        "",
        "Группы: A = RR/ритм, B = предсердная активность, C = time-frequency, "
        "D = морфология/организация, E = межотведенческая синхронизация.",
        "",
        "## Результаты (Chapman test)",
        "",
        "| Эксперимент | Модель | Признаков | F1-macro | AFIB-F1 | AFL-F1 | ROC-AUC | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.sort_values(["experiment", "model"]).iterrows():
        lines.append(f"| {r['experiment']} ({r['groups']}) | {r['model']} | {r['n_features']} | "
                     f"{r['chapman_test_f1_macro']:.3f} | {r['chapman_test_afib_f1']:.3f} | "
                     f"{r['chapman_test_afl_f1']:.3f} | {r['chapman_test_roc_auc']:.3f} | "
                     f"{r['chapman_test_brier']:.3f} |")
    lines += ["", "## Внешние наборы (порог из Chapman val)", "",
              "| Эксперимент | Модель | PTB-XL F1 / AUC | CPSC F1 / AUC |",
              "|---|---|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['experiment']} | {r['model']} | "
                     f"{r.get('ptbxl_f1_macro', float('nan')):.3f} / "
                     f"{r.get('ptbxl_roc_auc', float('nan')):.3f} | "
                     f"{r.get('cpsc2018_extra_f1_macro', float('nan')):.3f} / "
                     f"{r.get('cpsc2018_extra_roc_auc', float('nan')):.3f} |")
    lines += ["", "## Статистическая проверка (vs E0)", ""]
    for model, rows in boot.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Сравнение | ΔF1 | 95% CI | p | BH q | значимо |")
        lines.append("|---|---:|---|---:|---:|---|")
        for r in rows:
            lines.append(f"| {r['comparison']} | {r['mean']:+.4f} | "
                         f"[{r['lo']:+.4f}; {r['hi']:+.4f}] | {r['p_value']:.3f} | "
                         f"{r['bh_q']:.3f} | {'да' if r['significant'] else 'нет'} |")
        lines.append("")
    lines += ["## Лучшая конфигурация", "",
              json.dumps(best, indent=2, ensure_ascii=False, default=float), "",
              "## Ограничения", "",
              "- Прирост на внутреннем test не доказывает переносимость: внешние ROC-AUC "
              "остаются низкими (см. Stage 3.6.9).",
              "- F1 на внешних наборах зависит от их priors (AFL 3.6% в PTB-XL).",
              "- Locked holdout не использовался при выборе лучшей конфигурации.",
              ""]
    OUT1.write_text("\n".join(lines), encoding="utf-8")

    imp = ROOT / "reports" / "metrics" / "stage3610" / "feature_importance.csv"
    lines2 = ["# Stage 3.6.10 — Feature importance (SHAP)", "",
              "Импортанс рассчитан на зафиксированной лучшей модели; каждая строка содержит "
              "физиологическую группу признака.", ""]
    if imp.exists():
        fi = pd.read_csv(imp)
        lines2 += ["| Признак | Группа | importance |", "|---|---|---:|"]
        for _, r in fi.head(40).iterrows():
            lines2.append(f"| {r['feature']} | {r['group']} | {r['importance']:.5f} |")
        phys = MET / "physiology_direction_check.csv"
        if phys.exists():
            pc = pd.read_csv(phys)
            lines2 += ["", "## Проверка направления (Mann–Whitney U, AFL vs AFIB)", "",
                       "| Признак | Медиана AFL | Медиана AFIB | Направление | p |",
                       "|---|---:|---:|---|---:|"]
            for _, r in pc.iterrows():
                lines2.append(f"| {r['feature']} | {r['median_afl']:.4f} | "
                              f"{r['median_afib']:.4f} | {r['direction']} | {r['p_value']:.2g} |")
            lines2 += [
                "",
                "**Критический результат:** у трёх ведущих по SHAP признаков `periodicity` "
                "значения статистически **выше при AFIB**, а не при AFL (p < 1e-5), "
                "т.е. направление противоположно наивной физиологической гипотезе "
                "«AFL = более организованные F-волны».Возможные объяснения: "
                "(1) метрика periodicity измеряет не ту периодичность, которую мы ожидали "
                "(автокорреляция RR-подобной последовательности остатка); "
                "(2) в выборке Chapman много AFL с вторичным AFIB-кодом и наоборот; "
                "(3) сильный доменный/prior сдвиг. Направление **не вшивалось в классификатор** — "
                "модель сама использует признак, но интерпретация «AFL organized» "
                "для этого датасета не подтверждена.",
            ]
        else:
            lines2 += ["", "Проверка направления ещё не запущена."]
    else:
        lines2.append("SHAP-анализ ещё не запущен.")
    OUT2.write_text("\n".join(lines2), encoding="utf-8")
    print("Wrote", OUT1, OUT2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
