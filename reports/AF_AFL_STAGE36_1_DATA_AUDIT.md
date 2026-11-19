# Stage 3.6.1 — аудит пересобранных данных

Записей: 8962; форма: [8962, 12, 5000]; verdict: **ok**.

## Источник × класс × split

split                 test  train   val
source         label                   
chapman        0      1838   1841  1841
               1       592    592   593
cpsc2018_extra 0        15     15    14
               1        24     25    26
ptbxl          0        18     19    19
               1       496    497   497

Дубли signal_hash: 0; внутри split: 0.
patient_key в нескольких split: 0; PTB-XL patient_id в нескольких split: 0.

## Амплитуды

                     n  median_std_mv       p05       p95
chapman         7297.0       0.250139  0.145759  0.507204
cpsc2018_extra   119.0       0.236592  0.141761  0.458629
ptbxl           1546.0       0.229792  0.133503  0.498321

## SHA256

- `X.npy`: `faae7555005d8a1130e8736e31ef27d09f68fc1e693ece0016cc19d024613aeb`
- `meta.csv`: `349d334c12d63eb595c5f2d89fd5ed3fdfa37481a570cf3da95c9e2a2de3e3bb`
- `split.csv`: `8f3d4ad04a4108677c26dcb13ef2a77e921873554d2a446fe16eaf37bc028a4a`
- `qc_final_manifest.csv`: `526d7985858c43da289d8ea0c3f7531fef925e250f559b4d35b633656f02041c`
- `features_advanced_expert.npy`: `f3a29d62ea8919a48b53a98e6072b8a00d91003a5e1c12510a80ce6589791d10`
- `features_advanced_v2.npy`: `f92fc257dc036596bcb143174ac345e7ab274596dd7780dcf22e9b49209fd471`
- `features_heat.npy`: `d39f06ca608d7685bc1cd79dc1a7412dd08bd5c97fe2189ca076ddb005845205`
- `features_atrial34.npy`: `78837f4dbac3ee713f63a3cc3a679d598e64c396400e95562b4bc9e46862d0f4`