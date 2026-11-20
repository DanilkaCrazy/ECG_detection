"""Stage 3.6.18 — demote the over-used holdout to development status.

Stage 3.6.16 evaluated 4 feature sets x 3 seeds and Stage 3.6.17 re-used the
same split; it is therefore a DEVELOPMENT holdout, not an independent test.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics" / "stage3618"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    changed = []
    for name in ("af_afl_stage3614", "af_afl_stage3614_v1_1", "af_afl_stage3616"):
        p = ROOT / "data" / "processed" / name / "split.csv"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if "holdout" not in text:
            continue
        p.write_text(text.replace("holdout", "development_holdout"), encoding="utf-8")
        changed.append(str(p.relative_to(ROOT)))

    manifest = {
        "renamed_split_files": changed,
        "status": "development_holdout",
        "reason": "used for model comparison in Stage 3.6.16 and again in 3.6.17",
        "consequence": "no independent final evaluation has been performed yet",
        "action_required": "create a NEW sealed external set for the single final evaluation",
    }
    (OUT / "holdout_status.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
