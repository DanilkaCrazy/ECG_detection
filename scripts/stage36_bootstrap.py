"""Stage 3.6.1 — paired cluster bootstrap for baseline model comparison."""
import json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
D = ROOT / "data" / "processed" / "af_afl_stage34"


def f1(y, p, thr=0.5):
    q = (p >= thr).astype(int)
    tp = int(((y == 1) & (q == 1)).sum()); fp = int(((y == 0) & (q == 1)).sum())
    fn = int(((y == 1) & (q == 0)).sum()); tn = int(((y == 0) & (q == 0)).sum())
    p1 = tp / (tp + fp) if tp + fp else 0.0
    r1 = tp / (tp + fn) if tp + fn else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0
    r0 = tn / (tn + fp) if tn + fp else 0.0
    f1a = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    f1b = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return (f1a + f1b) / 2


def main():
    meta = pd.read_csv(D / "meta.csv")
    keys = meta.patient_key.to_numpy()
    files = sorted(MET.glob("stage36_2_pred_*.csv"))
    if len(files) < 2:
        print("need at least two prediction files", flush=True)
        return
    preds = {}
    for f in files:
        d = pd.read_csv(f)
        name = f.stem.replace("stage36_2_pred_", "")
        preds[name] = d.set_index("record_id")
    common = None
    for d in preds.values():
        common = set(d.index) if common is None else common & set(d.index)
    common = sorted(common)
    rec_pos = {r: i for i, r in enumerate(meta.record_id)}
    groups = np.array([keys[rec_pos[r]] for r in common])
    y = np.array([int(preds[files[0].stem.replace("stage36_2_pred_", "")].loc[r, "label"]) for r in common])

    names = sorted(preds)
    base = names[0]
    probs = {n: np.array([preds[n].loc[r, "prob_afib"] for r in common]) for n in names}
    thrs = {n: float(np.median([preds[n].loc[r, "threshold"] for r in common])) for n in names}
    rng = np.random.default_rng(42)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    diffs = {n: [] for n in names if n != base}
    for _ in range(1000):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
        f0 = f1(y[idx], probs[base][idx], thrs[base])
        for n in diffs:
            diffs[n].append(f1(y[idx], probs[n][idx], thrs[n]) - f0)
    out = {}
    for n, v in diffs.items():
        out[f"{n}_minus_{base}"] = {
            "mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
            "significant": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0),
        }
    (MET / "stage36_2_paired_bootstrap.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
