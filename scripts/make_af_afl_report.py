"""Assemble a consolidated Russian-language AF/AFL analysis report from all artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MET = REPORTS / "metrics"


def md(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    d = df[cols]
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join([head, sep] + rows)


def main() -> int:
    L: list[str] = []
    add = L.append

    add("# Анализ данных и классификация AF / AFL")
    add("")
    add("## 1. Какие данные используем")
    add("")
    add("Единый манифест: **44 075** записей, 8 классов. Источники: Chapman-Shaoxing (9 628 AF/AFL, "
        "99.99% всех данных) и MIT-BIH (2 записи). Частоты: 500 Гц (основная) и 360 Гц (MIT-BIH). "
        "Длительность: 10 с у Chapman, до ~30 мин у MIT-BIH. 12 отведений. Сигналы в **мВ** "
        "(СКО 0.1–0.5), все данные приведены к 500 Гц / 12 отведениям / 2500 или 5000 отсчётов.")
    add("")
    add("| Класс | Записей | Доля |")
    add("|---|---:|---:|")
    for c, n in [("SB", 16553), ("NSR", 8109), ("AFL", 7849), ("ST", 7146),
                 ("AF", 1781), ("SA", 1606), ("SVT", 708), ("VT", 323)]:
        add(f"| {c} | {n:,} | {100*n/44075:.2f}% |")
    add("")
    add("### Дисбаланс AF / AFL")
    add("")
    add("**AF = 1 781 (18.5%), AFL = 7 849 (81.5%), соотношение 4.41 : 1** в пользу AFL. "
        "Ранее бинарная модель обучалась на 300+300=600 записях (~6% доступных данных) — "
        "главный резерв был именно в объёме выборки.")
    add("")
    add("### Пропуски и качество")
    add("")
    add("- `patient_id`: 99.99% пропусков (сплит по `record_id`);")
    add("- `path_dat`: 99.99% пропусков (Chapman в `.mat`); `path_mat`: 6 пропусков (MIT-BIH в `.dat`);")
    add("- QC 600 случайных AF/AFL: 1 запись с NaN → заполнены нулями, 0 нечитаемых, все валидные;")
    add("- во всём датасете 765 записей имели NaN (заполнены), 2 потребовали ресемплинга, 0 отбраковано.")
    add("")
    add("Визуализации: `reports/figures/eda/` (8 файлов: гистограмма классов, AF/AFL по источникам, "
        "распределения fs/длительности/длины, пропуски, качество сигналов).")
    add("")

    # full datasets
    for dname, label in [("af_afl_full", "5 секунд (2500 отсчётов)"),
                         ("af_afl_full10s", "10 секунд (5000 отсчётов)")]:
        p = config.PROCESSED_ROOT / dname / "stats.json"
        if not p.exists():
            continue
        import json
        st = json.loads(p.read_text(encoding="utf-8"))
        add(f"## Данные: {label}")
        add("")
        add(f"Валидных записей: {st['n_kept']:,} (AF {st['n_af']:,}, AFL {st['n_afl']:,}), "
            f"сплит train={st['n_train']:,} / val={st['n_val']:,} / test={st['n_test']:,}. "
            f"QC: NaN заполнены у {st['qc']['nan_filled']}, ресемплинг {st['qc']['resampled']}, "
            f"отбраковано {st['qc']['bad']}.")
        add("")
        add("Очистка: NaN/Inf → 0, ресемплинг в 500 Гц, паддинг/обрезка до нужной длины, "
            "12 отведений, отбраковка константных/клиппированных сигналов, "
            "group-aware сплит (по patient_id, где доступен).")
        add("")

    # features
    fp = MET / "af_afl_features" / "feature_discrimination.csv"
    if fp.exists():
        d = pd.read_csv(fp).head(15)
        add("## 2. Признаки (пики, морфология, спектр)")
        add("")
        add("Извлекаются **579 признаков (v1)** и **615 (v2)**: статистики и Hjorth по 12 отведениям, "
            "спектральные полосы (0.5–100 Гц), R-пики и RR-интервалы (RMSSD, pNN50, CV, энтропия, "
            "автокорреляция, Пуанкаре SD1/SD2), морфология пиков (ширина, вариабельность амплитуды), "
            "автокорреляция предсердной активности 3–12 Гц в отведениях II/III/aVF/V1, "
            "межотводные корреляции.")
        add("")
        add("Топ признаков по силе различия (Mann–Whitney; >0 — выше в AF):")
        add("")
        add(md(d, ["feature", "effect", "p"]))
        add("")
        add("Визуализации: `reports/figures/af_afl_features/` (скрипичные графики, рейтинг признаков), "
            "`reports/figures/af_afl_imbalance*/feature_importance.png`.")
        add("")

    # classic imbalance
    for name, title in [("af_afl_imbalance", "5 с, v1-признаки"),
                        ("af_afl_imbalance_af_afl_full10s_v2", "10 с, v2-признаки")]:
        f = MET / name / "imbalance_results.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).sort_values("f1_macro", ascending=False)
        add(f"## 3{'a' if '5 с' in title else 'b'}. Балансировка ({title})")
        add("")
        add("Стратегии: `base`, `class_weight`, `oversample`, `undersample`, `SMOTE`, `SMOTE+class_weight`; "
            "модели: LogReg, RandomForest, ExtraTrees, HistGB, KNN, SVC-RBF. "
            "Порог подбирался на валидации; F1/accuracy/MCC — на тесте.")
        add("")
        add("| Стратегия | Train AFL / AF |")
        add("|---|---|")
        add("| base, class_weight | 5 500 / 1 241 |")
        add("| oversample, SMOTE, SMOTE+CW | 5 500 / 5 500 |")
        add("| undersample | 1 241 / 1 241 |")
        add("")
        add("Лучшие 10 конфигураций:")
        add("")
        add(md(d.head(10), ["strategy", "model", "f1_macro", "f1_af", "f1_afl",
                            "accuracy", "mcc", "auc_roc", "auc_pr"]))
        add("")
        add("> Примечание: в таблице `auc_roc`/`auc_pr` вычислены на валидации "
            "(остальные метрики — на тесте). Корректные тестовые AUC для лучших моделей "
            "приведены в разделе «Ансамбль».")
        add("")
        add("Визуализации: `reports/figures/" + name + "/` (гистограммы баланса до/после, "
            "ROC, PR, матрицы ошибок, сравнение метрик, важность признаков, чувствительность к порогу).")
        add("")

    # selection
    f = MET / "af_afl_selected_af_afl_full10s_v2" / "selected_results.csv"
    if f.exists():
        d = pd.read_csv(f).sort_values("auc_roc", ascending=False)
        add("## 4. Отбор признаков (SelectKBest)")
        add("")
        add("Проверено k ∈ {50, 100, 200, 400} × 3 стратегии × 4 модели. "
            "Лучший AUC **0.649** (SVC, class_weight, k=100), лучший F1_macro **0.563** "
            "(LogReg, class_weight, k=100). Отбор признаков немного повысил AUC, но не F1.")
        add("")
        add(md(d.head(8), ["k", "strategy", "model", "f1_macro", "f1_af", "auc_roc", "auc_pr"]))
        add("")

    # DL
    f = MET / "af_afl_full_dl" / "full_dl_results.csv"
    if f.exists():
        d = pd.read_csv(f).sort_values("f1_macro", ascending=False)
        add("## 5. Нейросетевые подходы (полные данные)")
        add("")
        add("CNN1D и ResNet1D с тремя вариантами балансировки: `pos_weight` (BCE с весом позитивного "
            "класса + WeightedRandomSampler), `oversample` (дублирование minority), `focal` "
            "(Focal loss α=0.75, γ=2). Ранняя остановка по val AUC, порог по val F1.")
        add("")
        add(md(d, ["model", "variant", "f1_macro", "f1_af", "accuracy", "auc_roc", "auc_pr", "log_loss"]))
        add("")
        add("Лучший DL: CNN1D/pos_weight F1_macro **0.553**, AUC **0.615** — уступает классическим "
            "моделям с ручными признаками.")
        add("")

    # ensemble
    f = MET / "af_afl_ensemble" / "ensemble_results.csv"
    if f.exists():
        d = pd.read_csv(f).sort_values("f1_macro", ascending=False)
        add("## 6. Ансамбль")
        add("")
        add("Усреднение вероятностей лучшей модели 5 с (SVC), лучшей 10 с (SVC) и CNN1D. "
            "Простое и AUC-взвешенное усреднение. Порог подбирался на валидации.")
        add("")
        add(md(d, ["ensemble", "threshold", "f1_macro", "f1_af", "accuracy",
                   "auc_roc", "auc_pr", "mcc", "log_loss"]))
        add("")
        add("Лучший F1 и AUC у одиночной 10-секундной модели (тестовый AUC **0.657**, "
            "F1_macro **0.572**); ансамбль улучшил log_loss (0.559) и accuracy (0.716).")
        add("")

    # audit
    audit = REPORTS / "metrics" / "eda" / "data_audit.txt"
    if audit.exists():
        add("## 7. Аудит данных (дополнение)")
        add("")
        add("- Разметка проверена по `.hea`: `164889003` = AFIB/AF, `164890007` = AFL — **перепутывания классов нет**.")
        add("- Цифры 1780/445 из статьи относятся к подвыборке PhysioNet Challenge; в полной базе AFIB 1780, AFL 8060.")
        add("- В нашей задаче **меньшинство — AF (1779, 18.5%)**, AFL — большинство (7835, 81.5%).")
        add("- Найдено 16 точных дубликатов сигналов (3 в train/test, 2 в train/val) — удалены; "
            "итоговый датасет `af_afl_full10s_dedup` (9614 записей).")
        add("- SMOTE/oversampling и нормализация применялись только к train; порог — только на val.")
        add("")

    # expert features
    f = MET / "af_afl_expert" / "expert_results.csv"
    if f.exists():
        d = pd.read_csv(f).sort_values("f1_macro", ascending=False)
        add("## 8. Физиологические признаки и дедуплицированные данные")
        add("")
        add("Реализован модуль `expert_af_afl.py` с 67 признаками: группы A (RR-статистики, Sample/Approx "
            "Shannon entropy), B (спектр), C (вейвлет-подобные полосы), D (организация предсердной активности), "
            "E (AV-проведение: ratio предсердной/желудочковой частоты, отклонение RR от целых кратных "
            "предсердного цикла). Сравнены: expert-only (67), v2 (615), combined (682).")
        add("")
        add(md(d.head(10), ["features", "strategy", "model", "f1_macro", "f1_af", "f1_afl",
                            "af_recall", "afl_recall", "auc_roc", "auc_pr", "log_loss"]))
        add("")
        add("Лучший: **combined + HistGB + class_weight** — F1_macro 0.571, AF-F1 0.337, "
            "AUC 0.667, log_loss 0.475 (лучшая калибровка среди всех экспериментов).")
        add("")

    # hybrid
    f = MET / "af_afl_hybrid" / "hybrid_results.csv"
    if f.exists():
        d = pd.read_csv(f)
        add("## 9. Гибридная архитектура")
        add("")
        add("`ResNet1DSE` (сырая морфология) + `InceptionTime1D` (разные масштабы) + RR-MLP + "
            "экспертный MLP с attention-объединением. Обучение: 0.5–40 Гц, per-lead z-score, "
            "сбалансированные батчи, Focal Loss (α=0.75, γ=2), аугментации, AF — положительный класс.")
        add("")
        add(md(d, ["variant", "f1_macro", "f1_af", "f1_afl", "accuracy", "auc_roc", "auc_pr", "mcc"]))
        add("")

    add("## 10. Выводы")
    add("")
    add("1. Старые метрики (F1 0.72, AUC 0.75) были получены на тесте из **22 записей** — "
        "статистически ненадёжно. На репрезентативном тесте (1 445 записей) реальные метрики ниже.")
    add("2. Увеличение данных с 600 до 9 630 записей и обработка дисбаланса дали устойчивый "
        "AUC ≈ 0.63–0.66 и F1_macro ≈ 0.57; лучший AF-F1 ≈ 0.34.")
    add("3. Наиболее эффективны: **class_weight / SMOTE + порог, подобранный на валидации**, "
        "и 10-секундное окно с расширенными RR/морфологическими признаками.")
    add("4. DL на сырых сигналах не превзошёл классические модели с признаками.")
    add("5. AF и AFL в этом датасете объективно трудны для разделения (AUC ~0.65–0.67): "
        "признаки различаются умеренно (макс. эффект Mann–Whitney ~0.15).")
    add("6. F1 0.90+ на 10-секундных записях Chapman не достигается ни одной из описанных "
        "архитектур. Работы с F1 0.90–0.99 используют минутные окна Холтера и другую разметку; "
        "для такого уровня необходимы длинные записи с несколькими десятками сердечных циклов "
        "и внешняя валидация на второй базе.")
    add("")
    add("Все артефакты: `reports/figures/`, `reports/metrics/`, `reports/models/`; "
        "скрипты воспроизведения — `scripts/eda_af_afl.py`, `build_af_afl_full.py`, "
        "`extract_af_afl_features.py`, `train_af_afl_imbalance.py`, `train_af_afl_selected.py`, "
        "`train_af_afl_full_dl.py`, `eval_af_afl_ensemble.py`, `make_af_afl_report.py`.")

    out = REPORTS / "AF_AFL_ANALYSIS_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
