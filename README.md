> **АКТУАЛЬНАЯ ЗАДАЧА (Stage 3.6):** бинарная классификация **AFIB (1) vs AFL (0)**
> по 10-секундной 12-канальной ЭКГ (500 Гц). Источники: Chapman, CPSC2018-extra,
> PTB-XL. Описание 8-классовой задачи ниже — историческая часть проекта.
>
> **Ограничения, которые нельзя игнорировать:**
> - Chapman не имеет настоящего `patient_id`; group = surrogate `record_id`.
>   Заявлять patient-level validation для Chapman нельзя.
> - Источник и класс сильно связаны (source classifier accuracy 0.9993),
>   LOSO не показывает переноса (AFL-F1 = 0). domain generalization не доказана.
> - Все метрики Stage 3.1–3.5 недействительны: внешние сигналы были на 99.76%
>   нулевыми из-за transpose-ошибки WFDB. Данные пересобраны 2026-11-19.
> - Новый честный baseline (Stage 3.6.2): ResNet1D Macro-F1 0.639 / AFIB-F1 0.540;
>   Expert+RandomForest 0.623 / 0.508. Различия между моделями статистически
>   не значимы. F1 ≥ 0.90 — исследовательская цель, не полученный результат.
> - Stage 3.6.1: строгая проверка 20764 записей выявила 55 Chapman-записей с
>   пропущенными грудными отведениями. Ablation новых 427 atrial-признаков ещё
>   не запускалась. См. `reports/AF_AFL_STAGE36_1_REPORT.md`.
>   См. `reports/AF_AFL_STAGE36_P0_AUDIT.md`.
>
> **Актуальные точки входа:** `scripts/stage36_extract_atrial.py`,
> `scripts/stage36_data_audit.py`, `scripts/stage36_strict_integrity.py`,
> `scripts/stage36_baseline.py`, `scripts/stage36_baseline_dl.py`.
>
> **ВНИМАНИЕ:** разделы ниже («Ключевые особенности», «Текущие результаты»,
> «Данные», «Методология оценки») описывают **историческую 8-классовую задачу
> Stage 1–2** и не относятся к текущему бинарному эксперименту. Утверждения о
> строгом `patient-level split` недействительны: Chapman не имеет настоящего
> `patient_id` и использует суррогатную группировку по записям. Актуальные
> отчёты: `reports/AF_AFL_STAGE36_1_DATA_AUDIT.md`,
> `reports/AF_AFL_STAGE36_2_BASELINE.md`.


# ECG Detection: Research Pipeline for Arrhythmia Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Исследовательский пайплайн для автоматизированной обработки, анализа и классификации нарушений сердечного ритма по цифровым ЭКГ-сигналам. Проект является прототипом программно-аналитической платформы **«ЭЛТА-Кардио»**, разрабатываемой в рамках магистерской диссертации, и закладывает основу для воспроизводимого MLOps-контура в анализе биомедицинских сигналов.

---

## Содержание

- [О проекте](#о-проекте)
- [Ключевые особенности](#ключевые-особенности)
- [Текущие результаты](#текущие-результаты)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Данные](#данные)
- [Методология оценки](#методология-оценки)
- [Планы развития](#планы-развития)
- [Лицензия](#лицензия)
- [Контакты](#контакты)

---

## О проекте

Сердечно-сосудистые заболевания остаются ведущей причиной смертности в мире, а электрокардиография (ЭКГ) — базовый неинвазивный метод их диагностики. Однако интерпретация ЭКГ требует высокой квалификации и сопряжена с субъективными ошибками. Проект **ECG_detection** направлен на автоматизацию этого процесса с помощью методов машинного и глубокого обучения.

**Основная цель** — создать воспроизводимую исследовательскую инфраструктуру, которая позволит:
- унифицировать обработку гетерогенных ЭКГ-датасетов (разные форматы, частоты дискретизации, конфигурации отведений);
- корректно обучать и сравнивать классические и нейросетевые модели с разделением выборки по пациентам (`patient-level split`);
- документировать результаты экспериментов и обеспечивать их воспроизводимость.

Проект развивается в направлении модульной архитектуры с единым контрактом данных (`ECGRecord`), что позволит в будущем перейти к полноценной платформе с версионированием данных, моделей и экспериментов (MLOps).

---

## Ключевые особенности

- **8 классов аритмий (историческая задача Stage 1–2, не текущая):** AF, AFL, NSR, SA, SB, ST, SVT, VT. Текущая задача — только AFIB (1) vs AFL (0).
- **Групповое разделение (не patient-level):** для Chapman доступен только суррогатный `patient_key = record_id`; для PTB-XL есть настоящий `patient_id`. Заявлять patient-level validation для Chapman нельзя.
- **Сравнение классических и глубоких моделей** в едином протоколе: RandomForest, LogisticRegression, KNN на инженерных признаках; 1D-CNN и ResNet1D на сырых сигналах.
- **Физиологически осмысленная аугментация** (гауссов шум, масштабирование) с возможностью расширения (time-warping, инъекция дрейфа изолинии, маскирование отведений).
- **Полный отчётный контур:** матрицы ошибок, per-class метрики (precision, recall, F1, support), кривые обучения, JSON/CSV-отчёты.
- **Воспроизводимость:** фиксация seed, сохранение конфигураций и весов моделей.

---

## Текущие результаты

Эксперименты проведены на 8 классах аритмий с использованием стратифицированного разделения по пациентам. В таблице приведены лучшие результаты для каждой группы моделей.

### Классические модели (RandomForest, признаки)

| Конфигурация | F1-macro | Accuracy |
|---|---|---|
| RandomForest + aug (robust_median_mad) | **0.7509** | **0.7600** |
| RandomForest + aug (none) | 0.7487 | 0.7571 |

### Нейросетевые модели (сырые сигналы)

| Модель | Аугментация | F1-macro | Accuracy |
|---|---|---|---|
| ResNet1D | нет | **0.6684** | **0.6792** |
| ResNet1D | да | 0.6658 | 0.6750 |
| CNN1D | да | 0.5082 | 0.5208 |
| CNN1D | нет | 0.4826 | 0.4979 |

**Ключевые наблюдения:**
- RandomForest на инженерных признаках остаётся сильным бейзлайном (F1 ≈ 0.75), что объясняется малым объёмом данных для нейросетей.
- ResNet1D на сырых сигналах показывает конкурентоспособный результат (F1 ≈ 0.67) и **лучше справляется с классом VT** (F1 0.82 против 0.65 на признаках), улавливая морфологические паттерны.
- **AFL — самый сложный класс** (F1 ≈ 0.24–0.38) из-за морфологического сходства с AF и NSR, а также выраженного дисбаланса.

> Полные отчёты, матрицы ошибок и кривые обучения сохранены в `reports/metrics/` и `reports/figures/`.

---

## Установка

### Требования
- Python 3.10+
- PyTorch 2.0+ (с поддержкой CUDA рекомендуется)
- WFDB, NumPy, pandas, scikit-learn, matplotlib, seaborn

### Шаги

```bash
# 1. Клонировать репозиторий
git clone https://github.com/DanilkaCrazy/ECG_detection.git
cd ECG_detection

# 2. Создать виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

> Если файл `requirements.txt` отсутствует, установите основные пакеты вручную:
> ```bash
> pip install torch torchvision scikit-learn pandas numpy wfdb matplotlib seaborn tqdm
> ```

---

## Быстрый старт

### 1. Подготовка данных

Проект работает с WFDB-совместимыми датасетами (например, Chapman, PTB-XL, MIT-BIH). Убедитесь, что данные загружены и пути к ним указаны в конфигурационном файле (например, `config.yaml` или через переменные окружения).

### 2. Запуск полного пайплайна (пример)

```bash
# Обучение и оценка классических моделей
python train_classic.py --config configs/classic.yaml

# Обучение и оценка нейросетевых моделей
python train_neural.py --config configs/resnet1d.yaml

# Генерация отчётов и визуализаций
python generate_reports.py --experiment_name resnet1d_baseline
```

> Точные названия скриптов и параметров могут отличаться — ориентируйтесь на структуру проекта (см. ниже) и документацию в коде.

### 3. Просмотр результатов

Все артефакты сохраняются в структурированном виде:

```
reports/
├── models/          # Веса моделей (.pt, .pkl)
├── figures/         # Матрицы ошибок, кривые обучения, сравнения
└── metrics/         # JSON/CSV отчёты, per-class метрики
```

---

## Структура проекта

```
ECG_detection/
├── data/                  # Сырые и обработанные данные (не включены в репозиторий)
├── src/                   # Исходный код
│   ├── data/              # Загрузка, аудит, предобработка сигналов
│   ├── features/          # Извлечение инженерных признаков
│   ├── models/            # Архитектуры (CNN1D, ResNet1D, классические модели)
│   ├── training/          # Циклы обучения, функции потерь, аугментация
│   ├── evaluation/        # Метрики, матрицы ошибок, отчёты
│   └── utils/             # Вспомогательные утилиты (seed, логирование)
├── configs/               # Конфигурационные файлы экспериментов
├── scripts/               # Скрипты запуска пайплайнов
├── reports/               # Результаты экспериментов
├── notebooks/             # Исследовательские ноутбуки (EDA, прототипы)
├── requirements.txt
└── README.md
```

---

## Данные

Проект не распространяет ЭКГ-данные из-за лицензионных ограничений. Для воспроизведения экспериментов используйте открытые источники:

- **Chapman / Shaoxing** — 12-канальные клинические ЭКГ (8 классов ритма).
- **PTB-XL** — 12-канальные клинические ЭКГ с многоуровневой разметкой.
- **MIT-BIH Arrhythmia** — 2-канальные амбулаторные записи.

Все датасеты доступны через [PhysioNet](https://physionet.org/). После загрузки укажите пути в конфигурации.

> **Важно:** разметка приводится к единой таксономии из 8 классов; для VT и AFL может потребоваться дополнительная верификация SNOMED-кодов.

---

## Методология оценки

- **Разделение выборки:** по `patient_key` (суррогатный для Chapman); PTB-XL использует настоящий `patient_id`.
- **Основная метрика:** macro-F1 (чувствительна к редким классам).
- **Дополнительные метрики:** per-class precision, recall, F1, support; accuracy; log-loss.
- **Визуализация:** матрицы ошибок (с нормализацией по строкам и без), кривые обучения, сравнение моделей.
- **Воспроизводимость:** фиксация seed (Python, NumPy, PyTorch), сохранение конфигураций и версий библиотек.

---

## Планы развития

Проект является прототипом платформы «ЭЛТА-Кардио». Ближайшие направления развития:

1. **Переход к модульной архитектуре** с единым контрактом данных `ECGRecord` и разделением на сервисы (Data Ingestion, Signal Processing, Dataset Builder, Training Engine, Model Registry).
2. **Борьба с дисбалансом классов:** внедрение focal loss с адаптивными весами, физиологически осмысленной аугментации (time-warping, lead masking).
3. **Адаптивное представление отведений:** обучение на смешанных конфигурациях каналов, маскирование отсутствующих отведений (гипотезы H1–H4 диссертации).
4. **Перенос на Холтеровские записи:** Domain Adaptation для детекции VT на длительных записях (гипотеза H5).
5. **Расширение набора архитектур:** CNN-LSTM, Transformer, lead-aware Transformer.
6. **Интеграция MLOps-инструментов:** MLflow для трекинга экспериментов, DVC для версионирования данных.

---

## Лицензия

Проект распространяется под лицензией **MIT**. См. файл [LICENSE](LICENSE) для подробностей.

---

## Контакты

**Автор:** Данил Вишняков 
**GitHub:** [@DanilkaCrazy](https://github.com/DanilkaCrazy)  
**Тема диссертации:** «Разработка программно-аналитической платформы для классификации электрокардиографических сигналов с применением методов машинного и глубокого обучения»

По вопросам сотрудничества и обсуждения исследований — открывайте Issue или связывайтесь напрямую.

---

*README составлен на основе текущего состояния проекта и может обновляться по мере развития платформы.*


---

## AFIB vs AFL: исследования Stage 1–3.0

Отдельная бинарная задача — различение фибрилляции (AFIB, SNOMED 164889003) и
трепетания (AFL, SNOMED 164890007) предсердий.

| Этап | Что сделано | Лучший результат (независимый тест) |
|---|---|---|
| Stage 1 | Полные 9 630 записей Chapman, 6 стратегий балансировки × 6 моделей, DL-архитектуры | Macro-F1 0.572, AFIB-F1 0.324 |
| Stage 2 | Строгая разметка (#Dx), 67 физиологических признаков, HEAT-inspired, дедупликация | **Macro-F1 0.6094, AFIB-F1 0.4413, ROC-AUC 0.6785** |
| Stage 3.0 | Аудит данных, гармонизация, GroupKFold, bootstrap CI, SHAP, CatBoost/XGBoost/LightGBM | Chapman Macro-F1 ≈ 0.57; внешние тесты AFIB-F1 0.97 |

**Важные оговорки:**
- полная база Chapman+Shaoxing (45 152 записи) объединяет Chapman и Ningbo; в локальных данных Ningbo отсутствует;
- в бинарной задаче меньшинство — AFIB (18.5%), а не AFL;
- прирост от HEAT-признаков в Stage 2 **не статистически значим** при 5-кратном групповом CV (Wilcoxon p=0.44);
- добавление внешних данных выравнивает классы (53/47), но вызывает domain shift; для Chapman предпочтительна модель, обученная только на Chapman;
- CPSC2018 не содержит AFL и используется только для pretraining AFIB vs others.

### Отчёты

- `reports/AF_AFL_ANALYSIS_REPORT.md` — Stage 1
- `reports/AF_AFL_STAGE2_REPORT.md` — аудит разметки, HEAT, анализ ошибок
- `reports/AF_AFL_MERGED_REPORT.md` — внешние AF-источники и балансировка
- `reports/AF_AFL_STAGE3_REPORT.md` — CatBoost/XGBoost/LightGBM + Optuna + SHAP
- `reports/AF_AFL_STAGE3_DATA_AUDIT.md`, `AF_AFL_STAGE3_VALIDATION.md`,
  `AF_AFL_STAGE3_HARMONIZATION.md`, `AF_AFL_STAGE3_HEAT_PLAN.md` — Stage 3.0

### Воспроизведение Stage 3.0

```bash
python scripts/stage3_data_integrity.py
python scripts/stage3_validation_protocol.py
python scripts/stage3_statistical_validation.py
python scripts/stage3_atrial_experiment.py
python scripts/train_stage3_gbdt.py
```


---

## Stage 3.1/3.2 — единый аудированный протокол (актуально)

Единый набор: `data/processed/af_afl_stage31/` — 9 042 записи, собранные
строго из `manifest_final.csv` (SHA256 зафиксирован в `stats.json`).
SHDB-AF исключён (ECG1/ECG2 не мапятся в 12 отведений), CPSC2018 (0 AFL) —
только pretrain-only. Полный QC всех записей: `reports/AF_AFL_STAGE31_FULL_QC.md`.

Воспроизведение:

```bash
python scripts/build_stage31_dataset.py
python scripts/extract_af_afl_features.py   # expert, v2, heat через AF_AFL_OUT/AF_AFL_FEAT
python scripts/stage31_full_qc.py
python scripts/stage32_compare_models.py     # E1-E7, выбор по validation
```

Отчёты: `reports/AF_AFL_STAGE31_FULL_QC.md`,
`reports/AF_AFL_STAGE32_REPORT.md`.


---

## Stage 3.3 — исправленный протокол (актуально)

- `scripts/stage33_full_qc.py` — полный QC с PASS/REVIEW/REJECT (9042 записи);
- `scripts/stage33_extract_features.py` — воспроизводимое извлечение признаков + SHA256;
- `scripts/test_harmonization.py` — 11 юнит-тестов гармонизации;
- `scripts/stage33_compare_strict.py` — 4-way split (train/cal/thr/select/test), выбор модели только на independent validation, cluster bootstrap.

Отчёты: `reports/AF_AFL_STAGE33_DATA_INTEGRITY.md`,
`reports/AF_AFL_STAGE33_STATISTICAL_VALIDATION.md`.


---

## Stage 3.4 — atrial activity и целостность эксперимента (актуально)

- QC STATUS **реально** фильтрует набор: 9042 → 8962 (80 REVIEW/REJECT удалены из X/y/meta/features);
- `src/ecgdet/atrial_activity_v2.py`: QRST cancellation (template subtraction), 80 признаков
  предсердной активности (частота, периодичность, морфология, II/III/aVF/V1, межотведенческая согласованность);
- `scripts/test_atrial_red.py` + `test_stage34_red.py`: 17 тестов, все проходят;
- `scripts/stage34_compare_features.py`: 8 feature sets × 6 моделей, 4-way split, 4 варианта порога;
- `scripts/stage34_error_analysis.py`: FN/FP с record_id, probability/threshold curves, permutation importance;
- `scripts/stage34_domain_generalization.py`: leave-one-source-out.

Отчёты: `reports/AF_AFL_STAGE34_REPORT.md`, `AF_AFL_STAGE34_ERROR_ANALYSIS.md`,
`AF_AFL_STAGE34_DOMAIN_GENERALIZATION.md`.


---

## Stage 3.5 — Experimental Integrity & Advanced Atrial Signal Modeling (актуально)

P0: error analysis и baseline используют ОДИН test split (1492 записи, model+calibrator из Stage 3.4), record_id проверены, LOSO preprocessingtrain-only, bootstrap CI. Достоверный baseline: F1 0.728, AFIB-F1 0.607.

P1: `src/ecgdet/atrial_signal_analysis.py` — multi-lead QRS, QRST cancellation (median/adaptive/beat-aligned) с измеренным качеством, спектральные/временные/межотведенческие признаки (62). 27 unit-тестов.

P2: сравнение на одном split — atrial-признаки НЕ улучшили метрики (лучший expert+RF F1 0.7315).

P3: source classifier accuracy 0.9993 (source confounding), LOSO AFL-F1 = 0.

Отчёты: `reports/AF_AFL_STAGE35_DATA_AUDIT.md`, `AF_AFL_STAGE35_BASELINE.md`,
`AF_AFL_STAGE35_ERROR_ANALYSIS.md`, `AF_AFL_STAGE35_DOMAIN_AUDIT.md`.
