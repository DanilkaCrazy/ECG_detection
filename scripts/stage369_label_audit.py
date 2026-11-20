"""Stage 3.6.9 (P0) — run the explicit AFIB/AFL label audit on the real dataset."""
from __future__ import annotations

import json
import sys
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.label_audit import audit_labels  # noqa: E402

D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "reports" / "metrics" / "stage369"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    result = audit_labels(meta)
    table = result.pop("table")
    table.to_csv(OUT / "label_audit_table.csv", index=False)
    # Counter objects are not JSON serialisable
    for src in result["by_source"].values():
        src["rhythms"] = {str(k): int(v) for k, v in src["rhythms"]}
    (OUT / "label_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "by_source"},
                     indent=2, ensure_ascii=False))
    print("\nBy source:")
    print(pd.DataFrame(result["by_source"]).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
