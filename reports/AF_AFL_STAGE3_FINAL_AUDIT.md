# Stage 3.0 — FINAL DATA AUDIT

Записей до дедупликации: **12292**, после: **12264** (удалено 28).

Конфликтов разметки у одинаковых сигналов: **0**.

## patient overlap

| source | records train/val/test | patients train/val/test | intersect tr-val, tr-te, val-te |
|---|---:|---:|---:|
| chapman | 2457/2458/2458 | 2457/2458/2458 | 0/0/0 |
| cpsc2018 | 0/0/0 | 0/0/0 | 0/0/0 |
| cpsc2018_extra | 40/41/41 | 40/41/41 | 0/0/0 |
| ptbxl | 516/516/515 | 422/422/421 | 0/0/0 |
| shdb-af | 708/729/704 | 30/23/24 | 0/0/0 |

Сигналов с хэшем в нескольких сплитах: **0**.

Статус: **PASS**.

Источники без обоих классов (`pretrain_only`): cpsc2018.

Ограничение: Chapman не имеет patient_id; group = record_id (одна запись = один пациент). PTB-XL group = patient_id.
