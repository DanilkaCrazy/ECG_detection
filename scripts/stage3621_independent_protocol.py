"""Stage 3.6.21 — prepare/freeze the protocol for a NEW independent test set.

No such set is currently available; this script FAILS CLOSED and documents
exactly what is required before the single final evaluation may be run.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAND = ROOT / "data" / "external" / "independent_test" / "meta.csv"
MET = ROOT / "reports" / "metrics" / "stage3621"
MET.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "reports" / "STAGE_3_6_21_INDEPENDENT_PROTOCOL.md"


def main() -> int:
    from sys import path
    path.insert(0, str(ROOT / "src"))
    from ecgdet.independent_evaluation import (  # noqa: E402
        independent_set_violations, freeze_artifacts, MIN_PER_CLASS,
    )
    if CAND.exists():
        meta = pd.read_csv(CAND)
        violations = independent_set_violations(meta)
    else:
        meta, violations = None, ["no independent test set has been supplied yet"]

    s17 = ROOT / "reports" / "metrics" / "stage3617" / "independent_assessment.json"
    thr = json.loads(s17.read_text(encoding="utf-8"))["threshold"] if s17.exists() else 0.4
    fn = pd.read_csv(ROOT / "data" / "processed" / "af_afl_stage3616" / "feature_names.csv")
    frozen = freeze_artifacts(
        ROOT / "reports" / "models" / "stage3617" / "frozen_E7.pt",
        fn.name.astype(str).tolist(), thr)

    report = {
        "status": "BLOCKED" if violations else "READY_FOR_SINGLE_FINAL_EVALUATION",
        "violations": violations,
        "requirements": {
            "min_per_class": MIN_PER_CLASS,
            "true_patient_id": True, "expert_labels": True,
            "sources": ">= 2 independent registration sites",
            "no_overlap_with_development": True,
        },
        "frozen_artifacts": frozen,
        "rule": "threshold/features/model must NOT be tuned on this set after freeze",
    }
    (MET / "independent_protocol.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    lines = [
        "# Stage 3.6.21 — Independent external evaluation protocol",
        "",
        f"**Статус: {report['status']}**. Никакой новой независимой выборки пока ",
        "не предоставлено; текущий development holdout использовался при сравнении моделей.",
        "",
        "## Требования к новому набору",
        "",
        f"- Не менее **{MIN_PER_CLASS}** записей каждого класса (AFIB/AFL).",
        "- Настоящие `patient_id`, экспертные метки, ≥2 источников регистрации.",
        "- Известные единицы, частота дискретизации, отсутствие пересечений ",
        "по `signal_hash` с development-данными.",
        "",
        "## Заморозка артефактов",
        "",
        json.dumps(frozen, indent=2, ensure_ascii=False),
        "",
        "После загрузки набора выполняется **одна** финальная оценка без подстройки.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
