# Stage 3.6.15 — Source Leakage & Signal Harmonization (P0)

Дата: 2026-11-20. Аудит performs on `af_afl_stage3614` (6180 записей с 
проверенным происхождением меток).

## 1. Физические характеристики источников

| Источник | n | RMS | p99 | zero-frac |
|---|---:|---:|---:|---:|
| chapman | 5973 | 0.278 | 1.290 | 0.0274 |
| cpsc2018_extra | 118 | 0.267 | 1.276 | 0.0065 |
| ptbxl | 89 | 0.225 | 1.033 | 0.0092 |

PTB-XL имеет заметно меньшую амплитуду (RMS 0.225 против 0.278 у Chapman). Это может отражать аппаратуру/фильтры, но не доказывает физиологическое различие.

## 2. Source classifier

| Эксперимент | Accuracy | Balanced Acc | ROC-AUC OVR |
|---|---:|---:|---:|
| A_metadata | 0.985 | 0.698 | 0.975 |
| B_raw_ecg | 0.985 | 0.684 | 0.988 |
| C_label_1 | 0.979 | 0.742 | — |
| C_label_0 | 0.989 | 0.667 | — |

Источник распознаётся по сигналу даже при ФИКСИРОВАННОМ классе: AFIB 0.979, AFL 0.989 accuracy. Значит различие источников не сводится к разной доле классов. Признаки построены только из сигнала — ID, пути и метки в них не попадают.

## 3. Исправленный UDA-протокол

Целевой домен разделён на adaptation (50%, без меток) и locked test (50%). Три режима Reported отдельно:

| Режим | Направление | Метод | Целевой AUC | Целевой F1 |
|---|---|---|---:|---:|
| inductive_uda | chapman→ptbxl | none | 0.058 | 0.171 |
| source_only | chapman→ptbxl | none | 0.058 | 0.171 |
| inductive_uda | chapman→ptbxl | CORAL | 0.088 | 0.306 |
| transductive_uda | chapman→ptbxl | CORAL | 0.119 | 0.168 |
| inductive_uda | chapman→ptbxl | MMD | 0.090 | 0.321 |
| transductive_uda | chapman→ptbxl | MMD | 0.119 | 0.191 |
| inductive_uda | chapman→ptbxl | DANN | 0.294 | 0.310 |
| transductive_uda | chapman→ptbxl | DANN | 0.558 | 0.349 |
| inductive_uda | chapman→cpsc2018_extra | none | 0.294 | 0.268 |
| source_only | chapman→cpsc2018_extra | none | 0.294 | 0.268 |
| inductive_uda | chapman→cpsc2018_extra | CORAL | 0.314 | 0.254 |
| transductive_uda | chapman→cpsc2018_extra | CORAL | 0.299 | 0.232 |
| inductive_uda | chapman→cpsc2018_extra | MMD | 0.321 | 0.254 |
| transductive_uda | chapman→cpsc2018_extra | MMD | 0.313 | 0.260 |
| inductive_uda | chapman→cpsc2018_extra | DANN | 0.368 | 0.322 |
| transductive_uda | chapman→cpsc2018_extra | DANN | 0.296 | 0.216 |
| inductive_uda | ptbxl→chapman | none | 0.489 | 0.370 |
| source_only | ptbxl→chapman | none | 0.489 | 0.370 |
| inductive_uda | ptbxl→chapman | CORAL | 0.488 | 0.363 |
| transductive_uda | ptbxl→chapman | CORAL | 0.497 | 0.360 |
| inductive_uda | ptbxl→chapman | MMD | 0.488 | 0.364 |
| transductive_uda | ptbxl→chapman | MMD | 0.497 | 0.365 |
| inductive_uda | ptbxl→chapman | DANN | 0.488 | 0.370 |
| transductive_uda | ptbxl→chapman | DANN | 0.498 | 0.378 |
| inductive_uda | cpsc2018_extra→chapman | none | 0.477 | 0.481 |
| source_only | cpsc2018_extra→chapman | none | 0.477 | 0.481 |
| inductive_uda | cpsc2018_extra→chapman | CORAL | 0.475 | 0.479 |
| transductive_uda | cpsc2018_extra→chapman | CORAL | 0.479 | 0.481 |
| inductive_uda | cpsc2018_extra→chapman | MMD | 0.476 | 0.477 |
| transductive_uda | cpsc2018_extra→chapman | MMD | 0.479 | 0.480 |
| inductive_uda | cpsc2018_extra→chapman | DANN | 0.468 | 0.482 |
| transductive_uda | cpsc2018_extra→chapman | DANN | 0.474 | 0.469 |

Inductive UDA на независимом целевом тесте не превысила 0.368 AUC (CPSC, DANN) и 0.294 (PTB-XL, DANN). transductive-режим формально выше, но eval'ируется на тех же записях, что и адаптация — это не доказательство обобщения.

## 4. Варианты предобработки (H0–H4)

| Вариант | F1 | AFIB-F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| H0_none | 0.516 | 0.306 | 0.530 | 0.254 |
| H1_per_lead_time | 0.510 | 0.275 | 0.518 | 0.255 |
| H2_robust | 0.521 | 0.292 | 0.538 | 0.278 |
| H3_interlead | 0.518 | 0.342 | 0.542 | 0.271 |
| H4_train_fitted | 0.515 | 0.277 | 0.524 | 0.249 |

## 5. Исправления preprocessing

- `stage3611_prepare.py`: простая децимация `sig[:, ::4]` заменена на `decimate_signal` (`resample_poly` с антиалиасинговой фильтрацией).
- `stage3611_train.py`, `stage3612_*.py`: нормализация по оси 1 (межотведенческая) заменена на нормализацию каждого отведения по времени.
- `stage3612_transfer.py`: preprocessing эмбеддингов теперь обучается только на train и применяется одинаково к val/test/external.
- Старые результаты сохранены (`modular_results.json`, `transfer_results.json`); новые — с суффиксом `_corrected`.

## 6. Ограничения

- H5 (сравнение 125/250/500 Гц) не завершено — требуется повторная экстракция признаков на трёх частотах.
- Алиас `AF → aVF` в `dataset_harmonization.py` остаётся непроверенным по документации конкретного источника; помечено как допущение.
- Экспертная ревизия меток ещё не выполнена (подготовлен только пакет).
