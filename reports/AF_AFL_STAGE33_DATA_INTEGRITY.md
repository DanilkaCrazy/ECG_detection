# Stage 3.3 — полный QC сигналов (PASS / REVIEW / REJECT)

Проверено **9042** записей.

                   n  PASS  REVIEW  REJECT  median_fs  median_n_samp  median_leads  max_abs_med
source                                                                                         
chapman         7373  7297      62      14      500.0         5000.0          12.0        2.577
cpsc2018_extra   122   119       1       2      500.0         5250.0          12.0        2.610
ptbxl           1547  1546       0       1      500.0         5000.0          12.0        2.291

PASS — все проверки пройдены; REVIEW — есть устранимые сомнения; REJECT — сигнал не пригоден. Только PASS используется в обучении.

Причины не-PASS:

- ('chapman', 'flat_leads=1'): 35.0
- ('chapman', 'flat_leads=6'): 8.0
- ('chapman', 'nan_frac=0.001'): 5.0
- ('chapman', 'nan_frac=0.000'): 5.0
- ('chapman', 'flat_leads=2'): 3.0
- ('chapman', 'amplitude>20mV:32.77'): nan
- ('chapman', 'amplitude>20mV:32.74'): nan
- ('chapman', 'amplitude>20mV:24.49'): nan
- ('cpsc2018_extra', 'flat_leads=1'): nan
- ('cpsc2018_extra', 'flat_leads=6'): 1.0
- ('cpsc2018_extra', 'nan_frac=0.001'): nan
- ('cpsc2018_extra', 'nan_frac=0.000'): nan
- ('cpsc2018_extra', 'flat_leads=2'): nan
- ('cpsc2018_extra', 'amplitude>20mV:32.77'): 1.0
- ('cpsc2018_extra', 'amplitude>20mV:32.74'): 1.0
- ('cpsc2018_extra', 'amplitude>20mV:24.49'): nan
- ('ptbxl', 'flat_leads=1'): nan
- ('ptbxl', 'flat_leads=6'): nan
- ('ptbxl', 'nan_frac=0.001'): nan
- ('ptbxl', 'nan_frac=0.000'): nan
- ('ptbxl', 'flat_leads=2'): nan
- ('ptbxl', 'amplitude>20mV:32.77'): nan
- ('ptbxl', 'amplitude>20mV:32.74'): nan
- ('ptbxl', 'amplitude>20mV:24.49'): 1.0

Правила: неизвестные единицы НЕ считаются mV; n_samp = shape[1]; NaN и Inf учитываются раздельно; амплитуда проверяется в mV после conversions.