# Stage 3.5 — анализ ошибок достоверного baseline

Модель all_heat_RandomForest, тест 1492 записей, порог 0.430.

CM [[912, 24], [303, 253]]; FN(AFIB)=303, FP(AFIB)=24.

Проверено: совпадение record_id между predictions/QC/features один-в-один; число строк совпадает с confusion matrix.

FN по источникам: {'chapman': 303}

FP по источникам: {'chapman': 9, 'cpsc2018_extra': 9, 'ptbxl': 6}

QC статус FN: {'PASS': 303}

Средняя надёжность QRST cancellation у FN (II/V1): {'atr_II_cancellation_reliable': 0.0, 'atr_V1_cancellation_reliable': 0.0}

Примеры FN (record_id, source, prob, periodicity, reliability):

record_id  source  prob_afib  atrial_periodicity_score  atr_II_cancellation_reliable
  JS00001 chapman   0.271839                       NaN                           0.0
  JS00007 chapman   0.197415                       NaN                           0.0
  JS00046 chapman   0.217987                       NaN                           0.0
  JS00092 chapman   0.226972                       NaN                           0.0
  JS00125 chapman   0.266371                       NaN                           0.0
  JS00151 chapman   0.286539                       NaN                           0.0
  JS00188 chapman   0.301602                       NaN                           0.0
  JS00234 chapman   0.264746                       NaN                           0.0
  JS00243 chapman   0.273914                       NaN                           0.0
  JS00249 chapman   0.239814                       NaN                           0.0
  JS00270 chapman   0.244291                       NaN                           0.0
  JS00272 chapman   0.228063                       NaN                           0.0
  JS00318 chapman   0.200033                       NaN                           0.0
  JS00335 chapman   0.350107                       NaN                           0.0
  JS00362 chapman   0.124919                       NaN                           0.0
  JS00393 chapman   0.267266                       NaN                           0.0
  JS00417 chapman   0.185336                       NaN                           0.0
  JS00421 chapman   0.234545                       NaN                           0.0
  JS00507 chapman   0.220784                       NaN                           0.0
  JS00589 chapman   0.314986                       NaN                           0.0