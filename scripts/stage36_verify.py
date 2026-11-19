"""Stage 3.6 P0 — post-extraction verification (NO training, NO selection)."""
import json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "metrics" / "af_afl_stage36"
PROC = ROOT / "data" / "processed" / "af_afl_stage36"


def main():
    diag = pd.read_csv(REP / "stage36_extraction_diagnostics.csv")
    feat = np.load(PROC / "features_atrial_residual.npy", mmap_mode="r")
    names = pd.read_csv(PROC / "feature_names_atrial_residual.csv")["name"].tolist()
    if "median_II_dominant_freq" in names:
        j = names.index("median_II_dominant_freq")
        diag["dom_freq"] = np.asarray(feat[:, j], dtype=float)
    else:
        diag["dom_freq"] = np.nan

    rows = []
    for src, g in diag.groupby("source"):
        rows.append({
            "source": src, "n": len(g),
            "qrs_peaks_median": float(g["n_qrs_peaks"].median()),
            "zero_qrs_frac": float((g["n_qrs_peaks"] == 0).mean()),
            "status_ok_frac": float((g["status_median"] == "ok").mean()),
            "qrs_supp_median": float(g["qrs_supp_median"].median()),
            "atr_pres_inside_median": float(g["atr_pres_inside_median"].median()),
            "dom_freq_median": float(g["dom_freq"].median()),
        })
    s = pd.DataFrame(rows)
    s.to_csv(REP / "stage36_quality_by_source.csv", index=False)
    summary = {
        "n_records": len(diag), "n_features": feat.shape[1],
        "feature_nan_fraction": float(np.isnan(np.asarray(feat)).mean()),
        "by_source": s.to_dict(orient="records"),
        "verdict": "P0 extraction usable" if (s["status_ok_frac"] > 0.5).all()
                   and (s["zero_qrs_frac"] < 0.05).all() else "NEEDS INVESTIGATION",
    }
    (REP / "stage36_verification.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
