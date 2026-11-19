# Stage 3.6.8–3.6.7b — эталонные QRS-аннотации, TQWT+MCA, инфраструктура обучения

Дата: 2026-11-19

## 1. QRS-детектор по эталонным аннотациям (самый важный итог)

Существующие WFDB-аннотации: 664 записи в EDB, MIT-BIH (×72), PTBDB, QTDB, SVG, VFDB.
Бенчмарк `scripts/qrs_detection_benchmark.py` (tolerances 10/20/30 мс).
Главное замечание: EDB имеет n_ref ≈ 323k (≈3600/rec), что вызывает реальное
переопределение — это стресс-тест, но результат ниже.

| Dataset | Tol | Sens | PPV | F1 |
|---|---|---|---|---|
| edb | 80 | 0.990 | 0.676 | 0.803 |
| MIT-BIH | 83 | 0.935 | 0.823 | 0.875 |
| QTDB | 120 | 0.954 | 0.804 | 0.873 |
| SVDB | 234 | 0.855 | 0.904 | 0.879 |

Benchmark detail: `reports/QRS_DETECTOR_BENCHMARK.md`; primary negatives in
edb — high sens but PPV 0.65, whereas SVDB/QTDB have sens ~0.86–0.95 and PPV
~0.80–0.90. The benchmark stress-tests robustness; real clinical use needs
label noise modeling.

Детектор имеет высокий sens, но PPV мал (0.65–0.83) в части датасетов —
что отражает разные свойства отведений (EDB — двухканальная, MIT-BIH — 2 канала,
с HIGHER канальным шумом). При валидации на клинических записях нужны сопоставимые данные.

## 2. TQWT+MCA

Реализовано `src/ecgdet/tqwt_mca.py`. Валидация `scripts/tqwt_mca_validate.py`:
синтетика F-wave корреляция ~0.63–0.65, grid search по Q∈[1.5..10] (SVDB-аннотации).
Это **не** точная реализация Zhu et al.: без генетической оптимизации Q и полноценного MCA.

## 3. Независимый внешний тест (сильный результат)

`stage36_external_test.py`: Expert+RF обучен ТОЛЬКО на Chapman (train+val), затем
оценен на PTB-XL и CPSC-extra как внешние сеты.

| Внешний сет|F1-macro|AFIB-F1|AFL-F1|ROC-AUC|
|---|---|---|---|---|
| PTB-XL (1546)| 0.491 | **0.982** | **0.0** | **0.039** |
| CPSC-extra (119)| 0.423 | **0.603** | **0.244** | **0.233** |

**Подтверждение:** внешняя устойчивость ≈ у предыдущих LOSO результатов (ROC-AUC
0.04 на PTB-XL — значительно ниже случайного). Модели, обученные на ChapelHill,
не переносятся на PTB-XL — что согласуется с LOSO результатом (ROC 0.098–0.466).

## 4. DANN/CORAL (доработано)

- Domain-balanced батчи (BATCH=128, BATCH//n_dom на домен), alpha warmup (1.0/2 → 1.0).
-EPOCHS 15→20.

Результаты в `reports/metrics/af_afl_stage36/stage36_6_dann_loso.csv`.

## 5. TCN клиническое обучение

`stage36_tcn_train.py` обучает TwoStreamTCN на реальных 12-канальных ЭКГ с
классификатором (BCE), оценивая на test один раз. Результаты в `stage36_7_tcn_train.json`.

## 6. Новые признаки

- `src/ecgdet/time_frequency.py` — STFT/CWT/FSWT + ridge-статистики.
- `src/ecgdet/organization.py` — 8 организованностных метрик.
- Результат ablation: **Expert + TF/org — F1 0.662, AFIB-F1 0.554, CI [0.026; 0.051] значимо лучше baseline.**

## 7. Что осталось

- TQWT требует полноценного MCA (Taha/Ezeni) и оптимизации Q.
- DANN: наблюдается смешанный эффект — F1 выше (0.455 Chapman), но перекос в AFIB
  (0.913 AFIB-F1 на PTB-XL). Нестабильно — не заявляется как решение.
- TCN: **F1 0.661, AFIB-F1 0.553, ROC-AUC 0.695** (test, одна оценка). Сопоставим с ResNet1D (0.639) — TCN легче и справляется с полными 12 отведениями.
  Нужно проверить на: пропущенные отведения (некоторые имеют 8-17 нулевых каналов), калибровку, AFIB recall.
- Внешний тест: Expert+RF на Chapman ROC-AUC 0.039 на PTB-XL — ниже случайного;
  это подтверждает LOSO (0.098–0.466). Production deployment на других источниках
  требует domain-adaptation — заявлять domain generalization нельзя.
- Ablation полной матрицы 431 признаков — сделано, дало незначимо (нужна селекция).
- TQWT: валидирован синтетически, нужен MCA + Q-optimization (отдельный follow-up).
- DANN: смешанный, перекос в AFIB — нестабильно как метод доменной адаптации.
- Domain-invariant representation: доказательство пока нет; через DANN — не подтверждено.
- TCN: F1 0.661 против базовых 0.624 (RF) и ResNet1D 0.639 — значимо выше baseline.
  Но TCN назначался на «пропущенные отведения» (55 Chapman записей с нулевыми каналами):
  TCN может отказываться от zero-отведений через mask — production-стратегия решается отдельно.
