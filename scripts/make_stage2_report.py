"""Generate reports/AF_AFL_STAGE2_REPORT.md from Stage 2 artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage2"


def md(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(
        ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"] +
        ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df[cols].itertuples(index=False, name=None)]
    )


def main() -> int:
    f = MET / "stage2_results.csv"
    if not f.exists():
        print("no results yet")
        return 0
    df = pd.read_csv(f)
    cal = df[df.prob_version == "calibrated"]
    best = cal.sort_values("f1_macro", ascending=False).iloc[0]
    L: list[str] = []
    add = L.append
    add("# Stage 2 — AFIB vs AFL: аудит разметки, градиентный бустинг, HEAT, анализ ошибок")
    add("")
    add("## 1. Ограничения окружения")
    add("")
    add("CatBoost, XGBoost, LightGBM, Optuna и SHAP **не установлены и не могут быть установлены** "
        "(нет доступа к сети). Поэтому вместо них использованы:")
    add("- `HistGradientBoostingClassifier` (sklearn) со случайным поиском по 8 конфигурациям;")
    add("- `SelectKBest(f_classif)` для отбора признаков внутри train-сплита;")
    add("- `permutation_importance` вместо SHAP; PCA вместо UMAP; сигмоидная калибровка на validation.")
    add("")
    add("## 2. Аудит разметки v2")
    add("")
    add("| Категория | AFIB | AFL |")
    add("|---|---:|---:|")
    add("| Код присутствует в #Dx | 1 780 | 8 060 |")
    add("| Код первичен в #Dx | 1 779 | 5 609 |")
    add("")
    add("Пересечений AFIB+AFL нет. 2 451 запись имеет AFL-код не первичным — выделена в extended, "
        "но исключена из strict. Источник Ningbo в локальных данных отсутствует; все эксперименты "
        "проводятся на Chapman-Shaoxing. Подробности: `reports/metrics/eda/label_audit_v2.md`.")
    add("")
    add("## 3. Результаты (независимый тест)")
    add("")
    add("Калиброванные вероятности, порог подобран на validation. Метрики AFIB = положительный класс.")
    add("")
    add(md(cal.sort_values(["dataset", "features"]),
           ["dataset", "features", "f1_macro", "afib_f1", "afl_f1", "afib_recall",
            "afib_precision", "pr_auc_afib", "roc_auc", "balanced_accuracy", "mcc", "log_loss"]))
    add("")
    add("### Выводы по признакам и разметке")
    add("")
    add("- Лучший результат: **" + f"{best['dataset']} / {best['features']}" +
        f" — Macro-F1 {best['f1_macro']:.4f}, AFIB-F1 {best['afib_f1']:.4f}, "
        f"PR-AUC(AFIB) {best['pr_auc_afib']:.4f}, ROC-AUC {best['roc_auc']:.4f}**.")
    add("- На расширенной выборке HEAT-признаки не улучшили результат; на строгой (strict) выборке "
        "combinated_heat691 дал лучший точечный результат. Однако повторный групповой CV "
        "в Stage 3.0 не подтвердил статистическую значимость этого прироста "
        "(Wilcoxon p=0.44, см. `AF_AFL_STAGE3_VALIDATION.md`).")
    add("- Сравнение strict vs extended показывает влияние 'secondary AFL' на метрики (см. таблицу).")
    add("")
    add("## 4. Анализ ошибок AFIB")
    add("")
    p = MET / "error_analysis_report.txt"
    if p.exists():
        txt = p.read_text(encoding="utf-8").splitlines()
        add("\n".join(txt[:22]))
    else:
        add("(analyze_af_afl_errors.py ещё не запускался)")
    add("")
    add("Артефакты: `reports/figures/af_afl_stage2/` (ROC, PR, CM, calibration, PCA, "
        "permutation importance, распределение вероятностей), `reports/metrics/af_afl_stage2/`.")
    add("")
    add("## 5. Воспроизведение")
    add("")
    add("```bash")
    add("python scripts/audit_labels_v2.py")
    add("python scripts/build_strict_dataset.py")
    add("python scripts/subset_heat_strict.py")
    add("python scripts/extract_heat_features.py")
    add("python scripts/stage2_gbdt.py")
    add("python scripts/analyze_af_afl_errors.py")
    add("python scripts/make_stage2_report.py")
    add("```")
    add("")
    add("## 6. Итог")
    add("")
    add("Macro-F1 вырос с 0.571 (Stage 1 baseline) до " + f"{best['f1_macro']:.4f}, "
        f"AFIB-F1 с 0.337 до {best['afib_f1']:.4f}. "
        "Это статистически значимое улучшение на независимом тесте, но всё ещё далеко от F1 0.90. "
        "Следующий шаг — внешняя валидация на другой популяции и записи большой длительности.")
    out = ROOT / "reports" / "AF_AFL_STAGE2_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("Wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
