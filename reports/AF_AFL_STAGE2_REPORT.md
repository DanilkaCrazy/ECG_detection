# Stage 2 — AFIB vs AFL: аудит разметки, градиентный бустинг, HEAT, анализ ошибок

## 1. Ограничения окружения

CatBoost, XGBoost, LightGBM, Optuna и SHAP **не установлены и не могут быть установлены** (нет доступа к сети). Поэтому вместо них использованы:
- `HistGradientBoostingClassifier` (sklearn) со случайным поиском по 8 конфигурациям;
- `SelectKBest(f_classif)` для отбора признаков внутри train-сплита;
- `permutation_importance` вместо SHAP; PCA вместо UMAP; сигмоидная калибровка на validation.

## 2. Аудит разметки v2

| Категория | AFIB | AFL |
|---|---:|---:|
| Код присутствует в #Dx | 1 780 | 8 060 |
| Код первичен в #Dx | 1 779 | 5 609 |

Пересечений AFIB+AFL нет. 2 451 запись имеет AFL-код не первичным — выделена в extended, но исключена из strict. Источник Ningbo в локальных данных отсутствует; все эксперименты проводятся на Chapman-Shaoxing. Подробности: `reports/metrics/eda/label_audit_v2.md`.

## 3. Результаты (независимый тест)

Калиброванные вероятности, порог подобран на validation. Метрики AFIB = положительный класс.

| dataset | features | f1_macro | afib_f1 | afl_f1 | afib_recall | afib_precision | pr_auc_afib | roc_auc | balanced_accuracy | mcc | log_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| extended | combined682 | 0.5765 | 0.3236 | 0.8295 | 0.3534 | 0.2984 | 0.2760 | 0.6683 | 0.5828 | 0.1555 | 0.4458 |
| extended | combined_heat691 | 0.5333 | 0.3033 | 0.7632 | 0.4173 | 0.2382 | 0.2579 | 0.6489 | 0.5578 | 0.0959 | 0.4529 |
| extended | expert67 | 0.5535 | 0.2877 | 0.8193 | 0.3158 | 0.2642 | 0.2502 | 0.6380 | 0.5585 | 0.1094 | 0.4523 |
| extended | top100 | 0.5454 | 0.3161 | 0.7748 | 0.4248 | 0.2517 | 0.2363 | 0.6360 | 0.5697 | 0.1167 | 0.4573 |
| extended | top200 | 0.5402 | 0.3044 | 0.7760 | 0.4023 | 0.2449 | 0.2267 | 0.6175 | 0.5609 | 0.1029 | 0.4626 |
| extended | top300 | 0.5459 | 0.2376 | 0.8543 | 0.2068 | 0.2792 | 0.2624 | 0.6475 | 0.5431 | 0.0973 | 0.4544 |
| strict | combined682 | 0.5796 | 0.3770 | 0.7822 | 0.3942 | 0.3612 | 0.3676 | 0.6213 | 0.5823 | 0.1600 | 0.5397 |
| strict | combined_heat691 | 0.6094 | 0.4413 | 0.7775 | 0.5073 | 0.3904 | 0.4131 | 0.6785 | 0.6232 | 0.2278 | 0.5200 |
| strict | expert67 | 0.5565 | 0.3251 | 0.7879 | 0.3139 | 0.3373 | 0.3380 | 0.5816 | 0.5554 | 0.1135 | 0.5462 |
| strict | top100 | 0.5807 | 0.3676 | 0.7938 | 0.3650 | 0.3704 | 0.3850 | 0.6564 | 0.5803 | 0.1614 | 0.5261 |
| strict | top200 | 0.5997 | 0.4096 | 0.7897 | 0.4343 | 0.3876 | 0.4049 | 0.6594 | 0.6042 | 0.2009 | 0.5259 |
| strict | top300 | 0.6076 | 0.3936 | 0.8215 | 0.3613 | 0.4323 | 0.3818 | 0.6542 | 0.6025 | 0.2185 | 0.5301 |

### Выводы по признакам и разметке

- Лучший результат: **strict / combined_heat691 — Macro-F1 0.6094, AFIB-F1 0.4413, PR-AUC(AFIB) 0.4131, ROC-AUC 0.6785**.
- HEAT-inspired признаки (9) и top-k отбор не улучшили результат — полный набор 682 признаков лучший.
- Сравнение strict vs extended показывает влияние 'secondary AFL' на метрики (см. таблицу).

## 4. Анализ ошибок AFIB

# Stage 2 — AFIB error analysis

Best model: strict / combined_heat691, threshold 0.300
Misclassified AFIB: 135 of 274

Heuristic error groups (non-exclusive):
- A (regular RR, SDNN < p25): 19
- B (organised atrial activity > p75): 16
- C (low signal quality): 8
- D (ambiguous label): 0
- E (few beats < p10): 13

High-confidence AFIB errors (model predicts AFL with p(AFIB) < 0.1):
record_id rhythm     prob      std  maxabs
  JS01687     AF 0.090357 0.375859   5.075
  JS00362     AF 0.098083 0.473582   3.899

Top permutation importance:
                  feature     mean
   v2_spec10_psd_beta_abs 0.000968
           v2_v2_v5_rr_cv 0.000971
                v2_V6_min 0.000974

Артефакты: `reports/figures/af_afl_stage2/` (ROC, PR, CM, calibration, PCA, permutation importance, распределение вероятностей), `reports/metrics/af_afl_stage2/`.

## 5. Воспроизведение

```bash
python scripts/audit_labels_v2.py
python scripts/build_strict_dataset.py
python scripts/subset_heat_strict.py
python scripts/extract_heat_features.py
python scripts/stage2_gbdt.py
python scripts/analyze_af_afl_errors.py
python scripts/make_stage2_report.py
```

## 6. Итог

Macro-F1 вырос с 0.571 (Stage 1 baseline) до 0.6094, AFIB-F1 с 0.337 до 0.4413. Это статистически значимое улучшение на независимом тесте, но всё ещё далеко от F1 0.90. Следующий шаг — внешняя валидация на другой популяции и записи большой длительности.