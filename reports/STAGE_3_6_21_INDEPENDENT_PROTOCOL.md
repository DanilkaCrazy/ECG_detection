# Stage 3.6.21 — Independent evaluation protocol

**Статус: BLOCKED**.

Причины блокировки:

- no independent test set has been supplied yet.

Требования: ≥100 записей каждого класса с экспертными метками, 
настоящие patient_id, ≥2 источников, отсутствие пересечений с development.

## Заморозка перед единственной финальной оценкой

{
  "model_path": "D:\\ИТМО\\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\\ECG_detection\\ECG_detection\\reports\\models\\stage3617\\frozen_E7.pt",
  "model_sha256": null,
  "feature_names_sha256": "98d2071110ba42a7d6994aa557a6b6a5f7153e5eceda34e48d1c5019d77b0e92",
  "n_features": 372,
  "threshold": 0.3999999999999999,
  "seed": 42,
  "note": "No tuning on the independent set after this freeze."
}

После заморозки запрещено подстраивать порог, признаки и preprocessing 
по меткам нового набора.
