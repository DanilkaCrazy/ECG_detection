# Stage 3.6.2 — новый baseline на пересобранных данных

train=2989, val=2990 (порог и калибровка), test=2983 (одна оценка).

Только expert-признаки (67), без новых atrial-признаков.

              model  f1_macro  afib_f1   afl_f1  afib_recall  roc_auc  pr_auc_afib    brier  threshold
expert_RandomForest  0.623004 0.507492 0.738517     0.472122 0.676818     0.602262 0.205518       0.39
      expert_HistGB  0.615645 0.505866 0.725424     0.484712 0.669426     0.598807 0.210058       0.40
    expert_CatBoost  0.619471 0.480169 0.758773     0.408273 0.670766     0.603088 0.209176       0.45