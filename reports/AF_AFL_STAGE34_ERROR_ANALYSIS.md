# Stage 3.4 — анализ ошибок

Тест: 2983 записей, порог 0.410 (зафиксирован на validation).

FN (пропущенные AFIB): 855, FP (ложные AFIB): 12

FN по источникам: {'chapman': 592, 'ptbxl': 250, 'cpsc2018_extra': 13}

FP по источникам: {'cpsc2018_extra': 7, 'ptbxl': 4, 'chapman': 1}

Примеры FN (20):
record_id  source  prob_afib status
  JS00007 chapman   0.172470   PASS
  JS00023 chapman   0.175699   PASS
  JS00030 chapman   0.134858   PASS
  JS00048 chapman   0.140574   PASS
  JS00050 chapman   0.186878   PASS
  JS00060 chapman   0.207982   PASS
  JS00085 chapman   0.207945   PASS
  JS00112 chapman   0.172846   PASS
  JS00125 chapman   0.199206   PASS
  JS00132 chapman   0.165499   PASS
  JS00176 chapman   0.181612   PASS
  JS00195 chapman   0.234816   PASS
  JS00208 chapman   0.142463   PASS
  JS00219 chapman   0.188086   PASS
  JS00249 chapman   0.178429   PASS
  JS00256 chapman   0.193649   PASS
  JS00272 chapman   0.197386   PASS
  JS00318 chapman   0.190726   PASS
  JS00330 chapman   0.168939   PASS
  JS00366 chapman   0.177708   PASS