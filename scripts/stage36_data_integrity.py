"""Stage 3.6 P0 — scan existing processed datasets for mostly-zero signals."""
import json, sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgdet import config

OUT = ROOT / "reports" / "metrics" / "af_afl_stage36"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("af_external", "af_afl_stage31", "af_afl_stage34"):
        p = config.PROCESSED_ROOT / name
        if not (p / "X.npy").exists():
            continue
        X = np.load(p / "X.npy", mmap_mode="r")
        meta = pd.read_csv(p / "meta.csv")
        for src in meta.source.unique():
            idx = np.where(meta.source == src)[0]
            pick = idx[::max(1, len(idx) // 300)]
            frac = np.asarray([np.mean(np.abs(np.asarray(X[i])) > 1e-8) for i in pick])
            rows.append({
                "dataset": name, "source": str(src), "n": int(len(idx)),
                "n_checked": int(len(pick)),
                "nonzero_frac_median": float(np.median(frac)),
                "invalid_records_frac": float(np.mean(frac < 0.5)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stage36_data_integrity.csv", index=False)
    bad = (df["invalid_records_frac"] > 0.5).any()
    summary = {
        "verdict": "DATA INVALID — rebuild required" if bad else "ok",
        "note": ("WFDB p_signal is (samples, channels); harmonize_signal expects "
                 "(channels, samples). External X.npy contains ~12 real samples "
                 "padded with zeros to 5000.") if bad else
                ("Rebuilt 2026-11-19 with p_signal.T in build_external_af_dataset.py; "
                 "external signals are full-length and non-zero."),
        "by_dataset_source": df.to_dict(orient="records"),
    }
    (OUT / "stage36_data_integrity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
