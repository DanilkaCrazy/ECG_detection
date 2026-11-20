# Stage 3.6.13 — кросс-датасетная генерализация

Направления: Chapman→PTB-XL, Chapman→CPSC-extra и обратно. Методы: без адаптации, CORAL, MMD, DANN. Адаптация использует только признаки целевого набора БЕЗ меток; порог выбран на source val и заморожен.

| Источник | Цель | Метод | AUC source-val | AUC target | F1 target |
|---|---|---|---:|---:|---:|
| chapman | ptbxl | none | 0.579 | 0.014 | 0.145 |
| chapman | ptbxl | CORAL | 0.583 | 0.018 | 0.209 |
| chapman | ptbxl | MMD | 0.583 | 0.016 | 0.221 |
| chapman | ptbxl | DANN | 0.572 | 0.023 | 0.178 |
| chapman | cpsc2018_extra | none | 0.579 | 0.263 | 0.210 |
| chapman | cpsc2018_extra | CORAL | 0.578 | 0.308 | 0.252 |
| chapman | cpsc2018_extra | MMD | 0.577 | 0.302 | 0.225 |
| chapman | cpsc2018_extra | DANN | 0.579 | 0.306 | 0.252 |
| ptbxl | chapman | none | 0.958 | 0.446 | 0.195 |
| ptbxl | chapman | CORAL | 0.990 | 0.432 | 0.195 |
| ptbxl | chapman | MMD | 0.989 | 0.430 | 0.195 |
| ptbxl | chapman | DANN | 0.992 | 0.439 | 0.207 |
| cpsc2018_extra | chapman | none | 0.896 | 0.471 | 0.418 |
| cpsc2018_extra | chapman | CORAL | 0.887 | 0.470 | 0.412 |
| cpsc2018_extra | chapman | MMD | 0.882 | 0.470 | 0.405 |
| cpsc2018_extra | chapman | DANN | 0.852 | 0.469 | 0.392 |

## Вывод

- **Ни один метод доменной адаптации не обеспечил перенос**: целевые ROC-AUC 0.014–0.471 при source-val AUC 0.58–0.99.
- CORAL/MMD дают небольшой прирост только на CPSC-extra (0.263→0.308), но не меняют качественную картину.
- DANN нестабилен и не исправляет перекос: Chapman→PTB-XL AUC 0.023.
- Обратный перенос PTB-XL→Chapman даёт AUC 0.446 (почти случайный), хотя внутри PTB-XL модель имеет AUC 0.96 — это прямое доказательство того, что модель учится source-specific признакам.

**Итог:** domain generalization по-прежнему не доказана. Увеличение сложности модели без исправления разметки/источников не решает проблему.
