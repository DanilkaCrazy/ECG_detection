# Stage 3.6.12 — перенос предварительно обученных представлений

## Preflight реального ECG-FM

- `fairseq` установлен: **False**
- Локальный чекпоинт: **None**
- Причина: fairseq backend is not installed (ECG-FM checkpoint is a fairseq model); no local ECG-FM checkpoint found
- ECG-FM на HuggingFace (`wanglab/ecg-fm`, MIT) существует: чекпоинт ~1.09 ГБ, архитектура `wav2vec2_cmsc` (24 слоя, embed_dim=1024) в формате fairseq. В этом окружении(torch 2.14+cpu, Python 3.14, без GPU) загрузка и инференс невыполнимы без установки fairseq — **реальный ECG-FM transfer не проведён**.

## Что реализовано вместо этого (SSL fallback)

Локальный masked-signal энкодер, предобучен на **2433** записях Chapman train только (без меток, без val/test), 8 эпох, mask_prob=0.4. Это НЕ ECG-FM.

| Подход | Chapman F1 | AFIB-F1 | ROC-AUC | PTB-XL AUC | CPSC AUC |
|---|---:|---:|---:|---:|---:|
| ssl_linear_probe | 0.517 | 0.286 | 0.524 | 0.298 | 0.443 |
| ssl_finetune | 0.477 | 0.197 | 0.506 | 0.467 | 0.521 |
| supervised_scratch | 0.496 | 0.280 | 0.519 | 0.428 | 0.439 |

## Статистическая проверка (парный кластерный бутстреп)

- **SSL fine-tune − scratch**: ΔF1 -0.0190, 95% CI [-0.0393; +0.0025], значимо = **False**.
- **Linear probe − scratch**: ΔF1 +0.0211, 95% CI [-0.0005; +0.0434], значимо = **False**.

## Вывод

- Результаты SSL нужно сравнивать с `supervised_scratch` на той же архитектуре: это единственный корректный способ измерить пользу предобучения.
- Даже при успешном SSL выход на внешние наборы остаётся ниже 0.5 — предобучение не устраняет domain shift.

## Как включить настоящий ECG-FM

1. Установить совместимый `fairseq` и загрузить `mimic_iv_ecg_physionet_pretrained.pt`.
2. Реализовать адаптер под `wav2vec2_cmsc` и снимать эмбеддинги с последнего слоя.
3. Повторить linear probing / fine-tuning на том же протоколе.
4. Отчитаться отдельно: `ecg_fm_frozen`, `ecg_fm_linear`, `ecg_fm_finetune`.
