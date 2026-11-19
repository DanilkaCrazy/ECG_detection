"""Stage 3.6.1 — strict data integrity: every record, every channel, hard gates."""
import json, sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet import config

FS_EXPECTED = 500
LEN_EXPECTED = 5000
N_LEADS_EXPECTED = 12
MIN_NONZERO = 0.5
MAX_NAN_FRAC = 0.0
MAX_INF_FRAC = 0.0


def check_record(x, *, fs=FS_EXPECTED, n_samp=LEN_EXPECTED, n_leads=N_LEADS_EXPECTED):
    """Return per-record integrity facts. Gate = all mandatory checks passed."""
    x = np.asarray(x)
    shape_ok = x.shape == (n_leads, n_samp)
    if x.ndim != 2:
        return {"shape_ok": False, "nonzero_min": np.nan, "nan_frac": np.nan,
                "inf_frac": np.nan, "amplitude_med_mv": np.nan, "gate": False,
                "reasons": "bad dimensionality"}
    nonzero = np.mean(np.abs(x) > 1e-8, axis=1)
    nan_frac = float(np.mean(np.isnan(x)))
    inf_frac = float(np.mean(np.isinf(x)))
    amp = float(np.median(np.abs(x)))
    reasons = []
    if not shape_ok:
        reasons.append("shape")
    if float(nonzero.min()) < MIN_NONZERO:
        reasons.append(f"channel_nonzero<{MIN_NONZERO}")
    if nan_frac > MAX_NAN_FRAC:
        reasons.append("nan")
    if inf_frac > MAX_INF_FRAC:
        reasons.append("inf")
    if amp < 1e-4 or amp > 5.0:
        reasons.append("amplitude_implausible")
    flat = np.where(nonzero < MIN_NONZERO)[0]
    return {"shape_ok": bool(shape_ok),
            "nonzero_min": float(nonzero.min()),
            "nonzero_median": float(np.median(nonzero)),
            "nan_frac": nan_frac, "inf_frac": inf_frac,
            "amplitude_med_mv": amp, "gate": not reasons,
            "n_flat_channels": int(len(flat)),
            "flat_channel_indices": ",".join(str(int(i)) for i in flat),
            "reasons": ";".join(reasons)}


def main():
    out = ROOT / "reports" / "metrics" / "af_afl_stage36"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("af_external", "af_afl_stage31", "af_afl_stage34"):
        p = config.PROCESSED_ROOT / name
        if not (p / "X.npy").exists():
            continue
        X = np.load(p / "X.npy", mmap_mode="r")
        meta = pd.read_csv(p / "meta.csv")
        src = meta.source.to_numpy()
        for i in range(len(X)):
            r = check_record(X[i])
            r.update({"dataset": name, "source": str(src[i]),
                      "record_id": str(meta.record_id.iloc[i]), "idx": i})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stage36_strict_integrity.csv", index=False)
    summary = {
        "n_checked": len(df),
        "n_per_dataset_source": {f"{a}/{b}": int(v) for (a, b), v in
                                 df.groupby(["dataset", "source"]).size().items()},
        "n_failed": int((~df["gate"]).sum()),
        "failed_by_dataset_source": {f"{a}/{b}": int(v) for (a, b), v in
                                     df[~df["gate"]].groupby(["dataset", "source"]).size().items()},
        "failed_reasons": df[~df["gate"]]["reasons"].value_counts().head(10).to_dict(),
        "verdict": "ok" if bool(df["gate"].all()) else "FAIL — mandatory checks violated",
    }
    (out / "stage36_strict_integrity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
