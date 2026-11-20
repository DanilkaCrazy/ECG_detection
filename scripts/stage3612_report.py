"""Stage 3.6.12 — foundation model transfer report (honest about ECG-FM blocker)."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics" / "stage3612"
OUT = ROOT / "reports" / "STAGE_3_6_12_FOUNDATION_TRANSFER.md"


def main() -> int:
    t = json.loads((M / "transfer_results.json").read_text(encoding="utf-8"))
    s = json.loads((ROOT / "reports" / "models" / "stage3612" / "ssl_pretrain.json").read_text(encoding="utf-8"))
    lines = [
        "# Stage 3.6.12 — перенос предварительно обученных представлений",
        "",
        "## Preflight реального ECG-FM",
        "",
        f"- `fairseq` установлен: **{s['ecg_fm_preflight']['fairseq_installed']}**",
        f"- Локальный чекпоинт: **{s['ecg_fm_preflight']['checkpoint']}**",
        f"- Причина: {s['ecg_fm_preflight']['reason']}",
        "- ECG-FM на HuggingFace (`wanglab/ecg-fm`, MIT) существует: чекпоинт ~1.09 ГБ, "
        "архитектура `wav2vec2_cmsc` (24 слоя, embed_dim=1024) в формате fairseq. "
        "В этом окружении(torch 2.14+cpu, Python 3.14, без GPU) загрузка и инференс "
        "невыполнимы без установки fairseq — **реальный ECG-FM transfer не проведён**.",
        "",
        "## Что реализовано вместо этого (SSL fallback)",
        "",
        f"Локальный masked-signal энкодер, предобучен на **{s['n_pretrain_records']}** "
        f"записях Chapman train только (без меток, без val/test), {s['epochs']} эпох, "
        f"mask_prob={0.4}. Это НЕ ECG-FM.",
        "",
        "| Подход | Chapman F1 | AFIB-F1 | ROC-AUC | PTB-XL AUC | CPSC AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("ssl_linear_probe", "ssl_finetune", "supervised_scratch"):
        b = t[name]["chapman_test"]
        lines.append(f"| {name} | {b['f1_macro']:.3f} | {b['afib_f1']:.3f} | "
                     f"{b['roc_auc']:.3f} | {t[name]['ptbxl']['roc_auc']:.3f} | "
                     f"{t[name]['cpsc2018_extra']['roc_auc']:.3f} |")
    lines += ["", "## Статистическая проверка (парный кластерный бутстреп)", ""]
    for key, label in [("bootstrap_ssl_finetune_minus_scratch", "SSL fine-tune − scratch"),
                       ("bootstrap_linear_probe_minus_scratch", "Linear probe − scratch")]:
        if key in t:
            b = t[key]
            lines.append(f"- **{label}**: ΔF1 {b['mean']:+.4f}, 95% CI "
                         f"[{b['lo']:+.4f}; {b['hi']:+.4f}], значимо = **{b['significant']}**.")
    lines += [
        "",
        "## Вывод",
        "",
        "- Результаты SSL нужно сравнивать с `supervised_scratch` на той же архитектуре: "
        "это единственный корректный способ измерить пользу предобучения.",
        "- Даже при успешном SSL выход на внешние наборы остаётся ниже 0.5 — "
        "предобучение не устраняет domain shift.",
        "",
        "## Как включить настоящий ECG-FM",
        "",
        "1. Установить совместимый `fairseq` и загрузить `mimic_iv_ecg_physionet_pretrained.pt`.",
        "2. Реализовать адаптер под `wav2vec2_cmsc` и снимать эмбеддинги с последнего слоя.",
        "3. Повторить linear probing / fine-tuning на том же протоколе.",
        "4. Отчитаться отдельно: `ecg_fm_frozen`, `ecg_fm_linear`, `ecg_fm_finetune`.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
