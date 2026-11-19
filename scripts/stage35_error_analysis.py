"""Stage 3.5 P0 — error analysis on the EXACT P0 baseline test predictions.

Uses stage35_test_predictions.csv from stage35_baseline_integrity.py —
the same model, same test indices, same threshold. Adds QC status, atrial
cancellation quality and key physiological features. No test-based selection.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage35"
FIG = ROOT / "reports" / "figures" / "af_afl_stage35"
D = ROOT / "data" / "processed" / "af_afl_stage34"
FIG.mkdir(parents=True, exist_ok=True)

THR = np.linspace(0.05, 0.95, 91)


def main() -> int:
    pred = pd.read_csv(MET / "stage35_test_predictions.csv")
    metrics = json.loads((MET / "stage35_metrics.json").read_text(encoding="utf-8"))
    man = pd.read_csv(D / "meta.csv")
    qc = pd.read_csv(ROOT / "reports" / "metrics" / "af_afl_stage33" / "full_signal_qc.csv")
    # merge by record_id — verify no row duplication/loss
    p2 = pred.merge(qc[["record_id", "status", "reasons", "flat_leads", "max_abs_mv"]],
                    on="record_id", how="left", validate="one_to_one")
    assert len(p2) == len(pred) == metrics["n_test"], "record_id merge mismatch"
    atrial_cols = [c for c in pd.read_csv(D / "feature_names_atrial34.csv")["name"]]
    feat = np.load(D / "features_atrial34.npy", mmap_mode="r")
    meta_ids = man.record_id.astype(str).to_numpy()
    pos = {rid: i for i, rid in enumerate(meta_ids)}
    idx = np.array([pos[str(r)] for r in p2.record_id])
    key_names = ["atrial_periodicity_score", "atrial_morphology_consistency",
                 "multilead_fwave_consistency", "atrial_rate_global",
                 "atr_II_qrs_removed_ratio", "atr_V1_qrs_removed_ratio",
                 "atr_II_cancellation_reliable", "atr_V1_cancellation_reliable"]
    for k in key_names:
        if k in atrial_cols:
            p2[k] = np.asarray(feat[idx, atrial_cols.index(k)], dtype=float)
    p2.to_csv(MET / "stage35_error_records.csv", index=False)

    fn = p2[p2.outcome == "FN"]
    fp = p2[p2.outcome == "FP"]
    lines = ["# Stage 3.5 — анализ ошибок достоверного baseline", "",
             f"Модель {metrics['model']}, тест {metrics['n_test']} записей, порог {metrics['threshold']:.3f}.", "",
             f"CM {metrics['cm']}; FN(AFIB)={len(fn)}, FP(AFIB)={len(fp)}.", "",
             "Проверено: совпадение record_id между predictions/QC/features один-в-один; "
             "число строк совпадает с confusion matrix.", "",
             "FN по источникам: " + str(fn.source.value_counts().to_dict()), "",
             "FP по источникам: " + str(fp.source.value_counts().to_dict()), "",
             "QC статус FN: " + str(fn.status.value_counts().to_dict()), "",
             "Средняя надёжность QRST cancellation у FN (II/V1): " +
             str(fn[["atr_II_cancellation_reliable", "atr_V1_cancellation_reliable"]].mean().to_dict()), "",
             "Примеры FN (record_id, source, prob, periodicity, reliability):", "",
             fn[["record_id", "source", "prob_afib", "atrial_periodicity_score",
                 "atr_II_cancellation_reliable"]].head(20).to_string(index=False)]
    (ROOT / "reports" / "AF_AFL_STAGE35_ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    # threshold curves (descriptive only, no selection)
    y = p2.label.to_numpy(); p = p2.prob_afib.to_numpy()
    rows = []
    for t in THR:
        pr = (p >= t).astype(int)
        rows.append({"threshold": t,
                     "f1_macro": f1_score(y, pr, average="macro"),
                     "afib_f1": f1_score(y, pr, pos_label=1),
                     "afib_recall": np.sum((pr==1)&(y==1))/max((y==1).sum(),1)})
    pd.DataFrame(rows).to_csv(MET / "stage35_threshold_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    d = pd.DataFrame(rows)
    for col in ("f1_macro", "afib_f1", "afib_recall"):
        ax.plot(d.threshold, d[col], label=col)
    ax.set_xlabel("threshold"); ax.legend(); ax.set_title("Descriptive threshold curves (test, not used for selection)")
    fig.tight_layout(); fig.savefig(FIG / "threshold_curves.png", dpi=150); plt.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
