# Stage 3.6.21 — Independent external evaluation protocol

**Статус: BLOCKED**. Никакой новой независимой выборки пока 
не предоставлено; текущий development holdout использовался при сравнении моделей.

## Требования к новому набору

- Не менее **100** записей каждого класса (AFIB/AFL).
- Настоящие `patient_id`, экспертные метки, ≥2 источников регистрации.
- Известные единицы, частота дискретизации, отсутствие пересечений 
по `signal_hash` с development-данными.

## Заморозка артефактов

{
  "model_path": "D:\\ИТМО\\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\\ECG_detection\\ECG_detection\\reports\\models\\stage3617\\frozen_E7.joblib",
  "model_sha256": "8f1b51484506115deaf736d6fc7f663820c90eaed74404f8169cf9c70721dc1a",
  "feature_names_sha256": "466428566268771fbb0f3314ec2e9f5b1917bf48ad1b2b8a987fba2251273a69",
  "n_features": 344,
  "threshold": 0.3999999999999999,
  "seed": 42,
  "note": "No tuning on the independent set after this freeze."
}

После загрузки набора выполняется **одна** финальная оценка без подстройки.
