# Stage 3.6.4 — LOSO и source-only baseline

held_out_source  n_train  n_test  afib_in_test  afl_in_test  threshold  f1_macro  afib_f1   afl_f1  roc_auc  pr_auc_afib preprocessing_fit_on
        chapman     1665    7297          1777         5520       0.51  0.198252 0.377785 0.018718 0.465787     0.229823   train sources only
 cpsc2018_extra     8843     119            75           44       0.41  0.601740 0.759036 0.444444 0.755758     0.844347   train sources only
          ptbxl     7416    1546          1490           56       0.28  0.280341 0.550329 0.010352 0.098682     0.895065   train sources only

Source-only (имя источника → класс): F1=0.7397, AFIB-F1=0.6246, AFL-F1=0.8547.

Chapman: patient_id недоступен, group = record_id (surrogate).
Порог выбран на train-источниках, test-источник не участвовал в настройке.