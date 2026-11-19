"""Stage 3.4 — leave-one-source-out domain generalization.

Train on two sources, test on the third. Uses fixed features (expert+v2+atrial),
HistGB, one global threshold chosen on a held-out part of TRAINING sources only.
Reports class counts per split — no strong conclusions if a class is scarce.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage34"
FIG = ROOT / "reports" / "figures" / "af_afl_stage34"

THR = np.linspace(0.05, 0.95, 91)


def main() -> int:
    man = pd.read_csv(D / "meta.csv")
    y = man.label.to_numpy()
    groups = man.patient_key.to_numpy()
    src = man.source.to_numpy()
    F = np.concatenate([np.load(D / f, mmap_mode="r") for f in
                        ("features_advanced_expert.npy", "features_advanced_v2.npy",
                         "features_atrial34.npy")], axis=1)
    an = np.isnan(F).all(axis=0)
    F = np.delete(F, np.where(an)[0], axis=1)
    med = np.nan_to_num(np.nanmedian(F, axis=0))
    F = np.where(np.isnan(F), med, F)
    F = np.nan_to_num(F, nan=-9, posinf=0, neginf=0).astype(np.float32)

    rows = []
    sources = sorted(set(src))
    for held_src in sources:
        tr = np.where(src != held_src)[0]
        te = np.where(src == held_src)[0]
        # threshold from a group-held-out part of TRAINING sources
        sg = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        fold = list(sg.split(tr, y[tr], groups=groups[tr]))[0]
        sub_tr, sub_thr = tr[fold[0]], tr[fold[1]]
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           max_leaf_nodes=20, min_samples_leaf=25,
                                           l2_regularization=0.5, class_weight="balanced",
                                           random_state=42)
        m.fit(F[sub_tr], y[sub_tr])
        pt = m.predict_proba(F[sub_thr])[:, 1]
        thr = float(THR[int(np.argmax([f1_score(y[sub_thr], (pt >= t).astype(int),
                                               average="macro") for t in THR]))])
        p = m.predict_proba(F[te])[:, 1]
        pred = (p >= thr).astype(int)
        r = {"held_out_source": held_src, "n_train": len(tr), "n_test": len(te),
             "afib_in_test": int((y[te] == 1).sum()), "afl_in_test": int((y[te] == 0).sum()),
             "threshold": thr,
             "f1_macro": float(f1_score(y[te], pred, average="macro")),
             "afib_f1": float(f1_score(y[te], pred, pos_label=1)),
             "afl_f1": float(f1_score(y[te], pred, pos_label=0)),
             "afib_recall": float(recall_score(y[te], pred, pos_label=1)),
             "afib_precision": float(precision_score(y[te], pred, pos_label=1)),
             "accuracy": float(accuracy_score(y[te], pred)),
             "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
             "mcc": float(matthews_corrcoef(y[te], pred)),
             "pr_auc_afib": float(average_precision_score(y[te], p)),
             "pr_baseline": float(y[te].mean()),
             "roc_auc": float(roc_auc_score(y[te], p))}
        rows.append(r)
        cm = confusion_matrix(y[te], pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns = __import__("seaborn"); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                                  xticklabels=["AFL", "AFIB"], yticklabels=["AFL", "AFIB"])
        ax.set_title(f"LOSO held-out={held_src}")
        fig.tight_layout(); fig.savefig(FIG / f"loso_{held_src}.png", dpi=150); plt.close()
        print(held_src, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "domain_generalization.csv", index=False)
    lines = ["# Stage 3.4 — leave-one-source-out", "",
             "Обучение на двух источниках, тест на третьем. Порог — только на части "
             "обучающих источников. Никакой source-specific калибровки на целевом домене.", "",
             df.to_string(index=False), "",
             "Ограничение: CPSC2018-extra содержит ~49 AFL; выводы по нему осторожные."]
    (ROOT / "reports" / "AF_AFL_STAGE34_DOMAIN_GENERALIZATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
