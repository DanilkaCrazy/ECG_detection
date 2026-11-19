# Stage 3.5 — DATA AUDIT

QC-manifest: 9042 записей -> 8962 PASS, исключено 80 ({'REJECT': 17, 'REVIEW': 63}).

Исправлено в P0:

- error analysis использует модель Stage 3.4 и ТОТ ЖЕ test split (1492 записи), что и baseline;

- проверка record_id один-в-один между predictions / QC / features;

- LOSO preprocessing (медианы, IQR, center) обучается только на train-источниках;

- предсказания и CM проверены на совпадение числа строк;

- bootstrap CI кластерный по patient_key.

Ограничение: Chapman не имеет настоящего patient_id; group = record_id (surrogate). Не заявлять patient-level validation для Chapman.

Source-label ассоциация (из AF_AFL_STAGE35_DOMAIN_AUDIT.md): source classifier accuracy 0.9993 — признаки кодируют источник; LOSO AFL-F1 = 0.

Базовая модель: all_heat_RandomForest, test 1492, F1=0.7277, AFIB-F1=0.6074.