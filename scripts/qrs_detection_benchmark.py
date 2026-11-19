"""Stage 3.6.8 — validate multi_lead_qrs against reference WFDB annotations."""
from __future__ import annotations
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import wfdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.atrial_signal_analysis import multi_lead_qrs

RAW = pathlib.Path("data/raw")
QRS_SYMS = {"N", "V", "R", "L", "S", "F", "A"}
DATASETS = ["edb", "mitbih", "ptbdb", "qtdb", "svdb", "vfdb"]


def match(found, ref, tol):
    """Two-pointer matching: returns (tp, fp, fn, mae_errors)."""
    f, r = sorted(found), sorted(ref)
    tp = 0; errors = []
    i = j = 0
    used_r = set()
    while i < len(f):
        # advance r pointer to >= f[i]-tol
        while j < len(r) and r[j] < f[i] - tol:
            j += 1
        k = j
        while k < len(r) and r[k] <= f[i] + tol:
            if k not in used_r:
                tp += 1; errors.append(abs(f[i] - r[k])); used_r.add(k); break
            k += 1
        i += 1
    fn = len(r) - len(used_r)
    return tp, len(f) - tp, fn, (np.mean(errors) if errors else 0.0), len(errors)


def _record_name(f: pathlib.Path) -> str | None:
    stem = str(f).replace("\\", "/").removesuffix(".atr")
    # wfdb may address segmented records as "<name>-N"; fall back gracefully.
    for cand in [stem]:
        if pathlib.Path(cand + ".hea").exists():
            return cand
    for cand in [stem.rsplit("-", 1)[0] if "-" in stem else stem]:
        if pathlib.Path(cand + ".hea").exists():
            return cand
    return None


def main():
    rows = []
    for ds in DATASETS:
        recs = sorted(RAW.glob(f"{ds}/**/*.atr"))[:40]
        for f in recs:
            name = _record_name(f)
            if name is None:
                continue
            rec = wfdb.rdrecord(name)
            ann = wfdb.rdann(name, "atr")
            ref = ann.sample[np.isin(ann.symbol, list(QRS_SYMS))]
            sig = np.asarray(rec.p_signal, dtype=float)
            if sig.ndim == 1:
                sig = sig[:, None]
            found, info = multi_lead_qrs(sig.T, float(rec.fs), list(rec.sig_name))
            for tol in (10, 20, 30):
                tp, fp, fn, mae, nev = match(found, list(ref), tol)
                rows.append({"dataset": ds, "record_id": f.stem, "fs": rec.fs,
                             "tol_ms": int(round(tol / rec.fs * 1000)),
                             "n_ref": len(ref), "n_found": len(found),
                             "tp": tp, "fp": fp, "fn": fn,
                             "sens": tp / len(ref) if len(ref) else np.nan,
                             "ppv": tp / (tp + fp) if tp + fp else np.nan,
                             "f1": (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else np.nan,
                             "mae_ms": mae * 1000 / rec.fs if mae else np.nan,
                             "n_qrs_votes": info.get("votes"), "det": info.get("qrs_detection_status")})
    df = pd.DataFrame(rows)
    out = pathlib.Path("reports/metrics/qrs_detector_benchmark.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Aggregate
    agg = df.groupby(["dataset", "tol_ms"]).agg(
        n_ref=("n_ref", "sum"), n_found=("n_found", "sum"), tp=("tp", "sum"),
        fp=("fp", "sum"), fn=("fn", "sum")).reset_index()
    agg["sens"] = agg["tp"] / agg["n_ref"]
    agg["ppv"] = agg["tp"] / agg["n_found"]
    agg["f1"] = 2 * agg["tp"] / (2 * agg["tp"] + agg["fp"] + agg["fn"])
    mae = df[df.tol_ms == 30].groupby("dataset")["mae_ms"].median()
    agg["mae_ms_median"] = agg["dataset"].map(mae)
    print(agg.round(4).to_string(index=False))
    agg.to_csv("reports/metrics/qrs_detector_benchmark_summary.csv", index=False)
    dup = df.groupby(["dataset", "tol_ms"])["fp"].mean()
    print("mean fp per record:", df[["dataset", "tol_ms", "fp"]].groupby(["dataset", "tol_ms"]).mean().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
