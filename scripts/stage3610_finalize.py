"""Stage 3.6.10 — write the full reproducible experiment directory."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "stage3610"
EXP = ROOT / "experiments" / "stage_3_6_10"


def main() -> int:
    best = json.loads((MET / "ablation_summary.json").read_text(encoding="utf-8"))["best"]
    d = EXP / f"{best['experiment']}_{best['model']}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True).strip()
    except Exception:
        git = "unknown"
    config = {
        "stage": "3.6.10",
        "experiment": best["experiment"],
        "groups": best["groups"],
        "model": best["model"],
        "seed": 42,
        "protocol": {
            "fit": "chapman/train", "threshold": "chapman/val",
            "control": "chapman/test", "external": ["ptbxl", "cpsc2018_extra"],
            "external_threshold_tuning": False,
        },
        "features": {
            "n": best["n_features"],
            "source": ["data/processed/af_afl_stage34/features_advanced_expert.npy",
                       "data/processed/af_afl_stage3610/X.npy"],
        },
        "git_commit": git,
    }
    (d / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in config.items()),
        encoding="utf-8")
    fi = MET / "feature_importance.csv"
    if fi.exists():
        shutil.copy(fi, d / "feature_importance.csv")
    (d / "run_metadata.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    (d / "logs").mkdir(exist_ok=True)
    print("Finalised", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
