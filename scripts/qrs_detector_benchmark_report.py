"""Stage 3.6.8 — QRS detector benchmark report."""
import pathlib, pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "metrics" / "qrs_detector_benchmark_summary.csv"
DET = ROOT / "reports" / "metrics" / "qrs_detector_benchmark.csv"
out = ROOT / "reports" / "QRS_DETECTOR_BENCHMARK.md"

agg = pd.read_csv(REP)
lines = ["# QRS-детектор benchmark на эталонных аннотациях", "",
         "Детектор: `multi_lead_qrs` из `atrial_signal_analysis.py` (робастный median+3MAD).",
         "Данные: " + ", ".join(sorted(agg["dataset"].unique())), "",
         agg.to_string(index=False), ""]
df = pd.read_csv(DET)
print(agg.to_string(index=False), flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
