# Stage 3.6.10 — Feature importance (SHAP)

Импортанс рассчитан на зафиксированной лучшей модели; каждая строка содержит физиологическую группу признака.

| Признак | Группа | importance |
|---|---|---:|
| B_median_III_periodicity | B | 0.01130 |
| B_adaptive_III_periodicity | B | 0.00786 |
| B_beat_aligned_III_periodicity | B | 0.00552 |
| B_median_II_atrial_cycle_cv | B | 0.00435 |
| B_adaptive_III_atrial_cycle_cv | B | 0.00408 |
| expert_55 | X | 0.00328 |
| B_beat_aligned_aVF_periodicity | B | 0.00287 |
| B_adaptive_III_peak_area_ratio_0_625hz | B | 0.00257 |
| B_median_III_atrial_cycle_cv | B | 0.00246 |
| B_beat_aligned_III_atrial_cycle_cv | B | 0.00241 |
| expert_22 | X | 0.00236 |
| B_adaptive_V1_peak_area_ratio_2_5hz | B | 0.00235 |
| expert_56 | X | 0.00231 |
| B_median_aVF_periodicity | B | 0.00223 |
| B_adaptive_V1_spectral_concentration | B | 0.00220 |
| B_median_III_peak_area_ratio_0_625hz | B | 0.00217 |
| B_adaptive_V1_cycle_morph_consistency | B | 0.00216 |
| B_tq_segment_III_spectral_flatness | B | 0.00213 |
| B_adaptive_II_atrial_cycle_cv | B | 0.00211 |
| B_adaptive_aVF_cycle_morph_consistency | B | 0.00206 |
| expert_57 | X | 0.00204 |
| expert_24 | X | 0.00201 |
| B_tq_segment_II_peak_area_ratio_2_5hz | B | 0.00201 |
| B_median_V1_spectral_concentration | B | 0.00199 |
| expert_23 | X | 0.00195 |
| B_beat_aligned_II_periodicity | B | 0.00193 |
| expert_37 | X | 0.00193 |
| expert_52 | X | 0.00190 |
| B_adaptive_II_peak_area_ratio_0_625hz | B | 0.00189 |
| expert_35 | X | 0.00188 |
| B_beat_aligned_aVF_peak_area_ratio_0_625hz | B | 0.00187 |
| B_adaptive_aVF_periodicity | B | 0.00186 |
| B_adaptive_V1_atrial_cycle_cv | B | 0.00186 |
| B_beat_aligned_III_peak_prominence | B | 0.00184 |
| expert_50 | X | 0.00182 |
| B_beat_aligned_V1_atrial_cycle_cv | B | 0.00180 |
| B_beat_aligned_V1_spectral_concentration | B | 0.00179 |
| B_beat_aligned_V1_cycle_morph_consistency | B | 0.00178 |
| expert_38 | X | 0.00172 |
| B_median_III_spectral_concentration | B | 0.00171 |

## Проверка направления (Mann–Whitney U, AFL vs AFIB)

| Признак | Медиана AFL | Медиана AFIB | Направление | p |
|---|---:|---:|---|---:|
| B_median_III_periodicity | 0.2354 | 0.2521 | AFIB> AFL | 5.2e-12 |
| B_adaptive_III_periodicity | 0.2273 | 0.2443 | AFIB> AFL | 1.2e-11 |
| B_beat_aligned_III_periodicity | 0.2439 | 0.2607 | AFIB> AFL | 2.6e-11 |
| B_median_II_atrial_cycle_cv | 0.3379 | 0.3281 | AFL> AFIB | 9.4e-12 |
| B_adaptive_III_atrial_cycle_cv | 0.3433 | 0.3368 | AFL> AFIB | 7.8e-08 |
| B_beat_aligned_aVF_periodicity | 0.2487 | 0.2691 | AFIB> AFL | 8.3e-15 |
| B_adaptive_III_peak_area_ratio_0_625hz | 0.5569 | 0.5844 | AFIB> AFL | 8.3e-11 |
| B_median_III_atrial_cycle_cv | 0.3409 | 0.3370 | AFL> AFIB | 3e-05 |
| B_beat_aligned_III_atrial_cycle_cv | 0.3412 | 0.3372 | AFL> AFIB | 0.00065 |
| B_adaptive_V1_peak_area_ratio_2_5hz | 0.7306 | 0.7239 | AFL> AFIB | 0.84 |

**Критический результат:** у трёх ведущих по SHAP признаков `periodicity` значения статистически **выше при AFIB**, а не при AFL (p < 1e-5), т.е. направление противоположно наивной физиологической гипотезе «AFL = более организованные F-волны».Возможные объяснения: (1) метрика periodicity измеряет не ту периодичность, которую мы ожидали (автокорреляция RR-подобной последовательности остатка); (2) в выборке Chapman много AFL с вторичным AFIB-кодом и наоборот; (3) сильный доменный/prior сдвиг. Направление **не вшивалось в классификатор** — модель сама использует признак, но интерпретация «AFL organized» для этого датасета не подтверждена.