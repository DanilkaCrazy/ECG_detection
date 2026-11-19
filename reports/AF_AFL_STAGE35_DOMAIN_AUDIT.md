# Stage 3.5 — LOSO с корректным preprocessing

Импутация, IQR-scaling и центрирование обучаются ТОЛЬКО на train-источниках.

held_out_source  n_train  n_test  afib_in_test  afl_in_test  threshold  f1_macro  afib_f1   afl_f1  afib_recall  afib_precision  accuracy  balanced_accuracy       mcc  pr_auc_afib  pr_baseline  roc_auc preprocessing_fit_on
        chapman     1665    7297          1777         5520       0.80  0.445528 0.031402 0.859653     0.016320        0.414286  0.754831           0.504446  0.039155     0.264437     0.243525 0.540584   train sources only
 cpsc2018_extra     8843     119            75           44       0.52  0.386598 0.773196 0.000000     1.000000        0.630252  0.630252           0.500000  0.000000     0.604380     0.630252 0.486970   train sources only
          ptbxl     7416    1546          1490           56       0.47  0.487570 0.975141 0.000000     0.987248        0.963327  0.951488           0.493624 -0.021625     0.962447     0.963777 0.469181   train sources only

Source classifier accuracy (group split): 0.9993

source x label:

col_0              0     1
row_0                     
chapman         5520  1777
cpsc2018_extra    44    75
ptbxl             56  1490

Chapman не имеет настоящего patient_id — group = record_id (surrogate).