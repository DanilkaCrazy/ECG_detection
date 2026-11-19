"""Final Stage 3 report: external AF integration + GBDT comparison + SHAP."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage3"


def md(df, cols):
    cols=[c for c in cols if c in df.columns]
    return "\n".join(["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]+
        ["| "+" | ".join(f"{v:.4f}" if isinstance(v,float) else str(v) for v in row)+" |"
         for row in df[cols].itertuples(index=False,name=None)])


def main():
    f=MET/"stage3_results.csv"
    if not f.exists():
        print("no results"); return 0
    df=pd.read_csv(f)
    st=json.loads((config.PROCESSED_ROOT/"af_afl_merged_v2"/"stats.json").read_text(encoding="utf-8"))
    L=[
        "# Stage 3 — внешние AF-данные, CatBoost/XGBoost/LightGBM + Optuna", "",
        "## 1. Сеть и библиотеки", "",
        "Повторная попытка установки прошла успешно: CatBoost, XGBoost, LightGBM, Optuna и SHAP доступны.", "",
        "## 2. Все найденные AF-источники", "",
        "| Источник | AFIB | AFL | Комментарий |",
        "|---|---:|---:|---|",
        "| Chapman-Shaoxing (strict #Dx) | 1 779 | 5 609 | основная база |",
        "| PTB-XL (AFIB/AFLT в scp_codes) | 1 514 | 73 | 12 отв., 500 Гц, 10 с |",
        "| CPSC2018 | 1 221 | 0 | 12 отв., 500 Гц |",
        "| CPSC2018-extra | 153 | 54 | 12 отв., 500 Гц |",
        "| SHDB-AF (аннотации `(AFIB`/`(AFL`) | 2 017 | 152 | 98 длинных записей, 2 отв. → 12 (pad) |",
        "| MIT-BIH / INCART / PTBDB | ~20 | ~5 | немного, разметка неоднозначна — в обучение не включены |",
        "",
        "Неоднозначные метки (оба кода AFIB+AFL в одной записи) отбрасывались. "
        "SHDB-окна брались только целиком внутри аннотированного интервала с отступом 5 с.", "",
        f"Итоговая объединённая выборка: **{st['n']:,}** записей, AFIB **{st['n_afib']:,}** "
        f"({100*st['n_afib']/st['n']:.1f}%), AFL **{st['n_afl']:,}** ({100*st['n_afl']/st['n']:.1f}%) — "
        "классы почти поровну.", "",
        "## 3. Результаты градиентного бустинга", "",
        "Калибровка и порог подбираются отдельно для каждого источника на validation; "
        "тесты не использовались при настройке.", "",
    ]
    for test in ("test_chapman","test_external","test_shdb"):
        L += [f"### {test}", "",
              md(df[df.test==test].sort_values("f1_macro",ascending=False),
                 ["model","f1_macro","afib_f1","afl_f1","afib_recall","afib_precision",
                  "pr_auc_afib","roc_auc","mcc","log_loss"]), ""]
    best=df[df.test=="test_chapman"].sort_values("f1_macro",ascending=False).iloc[0]
    L += ["## 4. Выводы", "",
          f"Лучшая модель на независимом Chapman-тесте: **{best['model']}** — "
          f"Macro-F1 {best['f1_macro']:.4f}, AFIB-F1 {best['afib_f1']:.4f}, "
          f"ROC-AUC {best['roc_auc']:.4f}.", "",
          "Экспертные признаки + HEAT + градиентный бустинг с Optuna дали ... (см. таблицу). "
          "SHAP-анализ сохранён в `reports/figures/af_afl_stage3/shap_summary.png`.", "",
          "Воспроизведение: `scripts/build_external_af_dataset.py`, `build_shdb_af_windows.py`, "
          "`merge_all_datasets.py`, `finish_merge_features.py`, `train_stage3_gbdt.py`."]
    out=ROOT/"reports"/"AF_AFL_STAGE3_REPORT.md"
    out.write_text("\n".join(L),encoding="utf-8")
    print("Wrote",out,flush=True)


if __name__=="__main__":
    main()
