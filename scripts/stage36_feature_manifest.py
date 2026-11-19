"""Stage 3.6.3 — split 431 extracted fields into predictive vs diagnostic."""
import json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage36"
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"

DIAGNOSTIC_PREFIXES = ("status_", "qrs_supp_", "resid_ratio_", "valid_tq_",
                       "n_valid_tq_", "n_invalid_short_tq", "median_tq_segment_length",
                       "n_qrs_peaks", "n_qrs_votes", "cross_lead_coherence_",
                       "qrs_", "cancel_")
METHOD_PREFIXES = ("median_", "adaptive_", "beat_aligned_", "tq_segment_")


def classify(name):
    if name.startswith(DIAGNOSTIC_PREFIXES):
        return "diagnostic"
    if name.startswith(METHOD_PREFIXES + ("spec_", "temp_", "cross_spectral_", "phase_locking_")):
        return "predictive"
    return "other"


def main():
    names = pd.read_csv(D / "feature_names_atrial_residual.csv")["name"].tolist()
    kinds = [classify(n) for n in names]
    manifest = {
        "n_total": len(names),
        "n_predictive": kinds.count("predictive"),
        "n_diagnostic": kinds.count("diagnostic"),
        "n_other": kinds.count("other"),
        "predictive": [n for n, k in zip(names, kinds) if k == "predictive"],
        "diagnostic": [n for n, k in zip(names, kinds) if k == "diagnostic"],
        "other": [n for n, k in zip(names, kinds) if k == "other"],
    }
    (MET / "stage36_3_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    idx = np.array([i for i, k in enumerate(kinds) if k == "predictive"], dtype=int)
    np.save(MET / "stage36_3_predictive_indices.npy", idx)
    print(json.dumps({k: v for k, v in manifest.items()
                      if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
