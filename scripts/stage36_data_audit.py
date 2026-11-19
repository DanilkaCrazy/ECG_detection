"""Stage 3.6.1 — full audit of the rebuilt dataset (no model training)."""
import hashlib, json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "af_afl_stage34"
OUT = ROOT / "reports" / "metrics" / "af_afl_stage36"
OUT.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    meta = pd.read_csv(D / "meta.csv")
    split = pd.read_csv(D / "split.csv")
    X = np.load(D / "X.npy", mmap_mode="r")
    assert len(meta) == len(split) == len(X) == 8962

    hashes = {p: sha(D / p) for p in
              ("X.npy", "meta.csv", "split.csv", "qc_final_manifest.csv")}
    for f in ("features_advanced_expert.npy", "features_advanced_v2.npy",
              "features_heat.npy", "features_atrial34.npy",
              "features_atrial_residual.npy"):
        if (D / f).exists():
            hashes[f] = sha(D / f)

    table = pd.crosstab([meta.source, meta.label], meta.split)
    dup_hash = int(meta.signal_hash.duplicated().sum())
    dup_hash_cross = int(meta.groupby("split")["signal_hash"].apply(
        lambda s: s.duplicated().sum()).sum())

    # patient_key overlap between splits (Chapman uses surrogate record grouping)
    key_splits = meta.groupby("patient_key")["split"].nunique()
    key_overlap = int((key_splits > 1).sum())
    ptb = meta[meta.source == "ptbxl"]
    pid_splits = ptb.groupby("patient_id")["split"].nunique() if "patient_id" in ptb else pd.Series(dtype=int)
    pid_overlap = int((pid_splits > 1).sum()) if len(pid_splits) else 0

    amp = {}
    for src, g in meta.groupby("source"):
        idx = np.where(meta.source == src)[0][::max(1, len(g) // 400)]
        vals = [float(np.std(np.asarray(X[i]))) for i in idx]
        amp[str(src)] = {"n": int(len(g)), "median_std_mv": float(np.median(vals)),
                         "p05": float(np.percentile(vals, 5)),
                         "p95": float(np.percentile(vals, 95))}

    summary = {
        "n_records": int(len(meta)), "shape": list(X.shape),
        "sha256": hashes,
        "source_label_split": table.reset_index().to_dict(orient="records"),
        "duplicate_signal_hashes": dup_hash,
        "duplicate_hashes_within_split": dup_hash_cross,
        "patient_key_in_multiple_splits": key_overlap,
        "ptbxl_patient_id_in_multiple_splits": pid_overlap,
        "grouping_note": "Chapman: no real patient_id, patient_key=record_id surrogate; "
                         "PTB-XL: patient_id present; CPSC2018-extra: record surrogate",
        "amplitude_by_source": amp,
        "fs_hz": 500, "n_samples": 5000,
        "verdict": "ok" if dup_hash == 0 and key_overlap == 0 and pid_overlap == 0
                   else "INVESTIGATE",
    }
    (OUT / "stage36_1_data_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage 3.6.1 — аудит пересобранных данных", "",
             f"Записей: {summary['n_records']}; форма: {summary['shape']}; verdict: **{summary['verdict']}**.", "",
             "## Источник × класс × split", "", table.to_string(), "",
             f"Дубли signal_hash: {dup_hash}; внутри split: {dup_hash_cross}.",
             f"patient_key в нескольких split: {key_overlap}; PTB-XL patient_id в нескольких split: {pid_overlap}.", "",
             "## Амплитуды", "", pd.DataFrame(amp).T.to_string(), "",
             "## SHA256", ""]
    for k, v in hashes.items():
        lines.append(f"- `{k}`: `{v}`")
    (ROOT / "reports" / "AF_AFL_STAGE36_1_DATA_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
