# Stage 3.6.2 — новый baseline на пересобранных данных

train=2989, val=2990 (порог и калибровка), test=2983 (одна оценка).

Только expert-признаки (67), без новых atrial-признаков.

              model  f1_macro  afib_f1   afl_f1  afib_recall  roc_auc  pr_auc_afib    brier  threshold
expert_RandomForest  0.623917 0.512845 0.734990     0.484712 0.676818     0.602262 0.205942       0.39
      expert_HistGB  0.615503 0.501430 0.729576     0.473022 0.669426     0.598807 0.210370       0.41
    expert_CatBoost  0.619770 0.481852 0.757688     0.411871 0.670766     0.603088 0.209228       0.45