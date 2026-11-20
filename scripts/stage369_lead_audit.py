"""Stage 3.6.9 (P0) — lead availability / signal quality audit on the dataset."""
from __future__ import annotations

import json
import sys
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet.lead_quality_audit import (  # noqa: E402
    CANONICAL_LEADS, lead_quality, summarize_dataset_leads,
)

D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "reports" / "metrics" / "stage369"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    meta = pd.read_csv(D / "meta.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    summary = summarize_dataset_leads(X, meta, 500.0, list(CANONICAL_LEADS))
    summary.to_csv(OUT / "lead_quality_by_source.csv", index=False)

    # Full per-record scan of usability (cheap: one FFT per lead per record).
    rows = []
    for i in range(len(meta)):
        q = lead_quality(np.asarray(X[i]), 500.0, list(CANONICAL_LEADS))
        rows.append({
            "record_id": meta.record_id.iloc[i],
            "source": meta.source.iloc[i],
            "label": int(meta.label.iloc[i]),
            "n_usable": int(q.usable.sum()),
            "usable_leads": ",".join(q.loc[q.usable, "lead"].astype(str)),
            "unusable_leads": ",".join(q.loc[~q.usable, "lead"].astype(str)),
        })
    per_rec = pd.DataFrame(rows)
    per_rec.to_csv(OUT / "lead_quality_per_record.csv", index=False)
    stat = {
        "n_records": len(per_rec),
        "n_with_all_leads": int((per_rec.n_usable == 12).sum()),
        "n_with_missing_leads": int((per_rec.n_usable < 12).sum()),
        "n_with_single_good_lead": int((per_rec.n_usable == 1).sum()),
        "n_with_zero_good_leads": int((per_rec.n_usable == 0).sum()),
        "mean_usable_leads": float(per_rec.n_usable.mean()),
        "by_source": per_rec.groupby("source")["n_usable"].agg(["mean", "min", "max"]).to_dict("index"),
    }
    (OUT / "lead_audit.json").write_text(
        json.dumps(stat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stat, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
