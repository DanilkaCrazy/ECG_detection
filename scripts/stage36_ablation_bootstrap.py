"""Stage 3.6.3 — paired cluster bootstrap for ablation configurations."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
MET = ROOT / "reports" / "metrics" / "af_afl_stage36"
D = ROOT / "data" / "processed" / "af_afl_stage34"
meta = pd.read_csv(D / "meta.csv")
keys = meta.patient_key.to_numpy()
pos = {r: i for i, r in enumerate(meta.record_id)}

files = sorted(MET.glob("stage36_3_pred_*_RF.csv"))
preds = {}
for f in files:
    name = f.stem.replace("stage36_3_pred_", "").replace("_RF", "")
    preds[name] = pd.read_csv(f).set_index("record_id")
common = set.intersection(*[set(d.index) for d in preds.values()])
common = sorted(common)
groups = np.array([keys[pos[r]] for r in common])
base = "A_expert"
y = preds[base].loc[common, "label"].to_numpy()
prob = {n: preds[n].loc[common, "prob"].to_numpy() for n in preds}
thr = {n: float(preds[n].loc[common, "prediction"].iloc[0] * 0 + 0.5) for n in preds}
# recover thresholds from original results
res = pd.read_csv(MET / "stage36_3_ablation.csv")
thr = {r["set"]: float(r["threshold"]) for _, r in res.iterrows() if r["model"] == "RF"}

def f1m(yy, p, t):
    return float(f1_score(yy, (p >= t).astype(int), average="macro"))

rng = np.random.default_rng(42)
ug = np.unique(groups); by = {g: np.where(groups == g)[0] for g in ug}
diffs = {n: [] for n in preds if n != base}
for _ in range(1000):
    idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
    f0 = f1m(y[idx], prob[base][idx], thr[base])
    for n in diffs:
        diffs[n].append(f1m(y[idx], prob[n][idx], thr[n]) - f0)
out = {}
for n, v in diffs.items():
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    out[f"{n}_minus_{base}"] = {"mean": float(np.mean(v)), "lo": lo, "hi": hi,
                                "significant": bool(lo > 0 or hi < 0)}
(MET / "stage36_3_ablation_bootstrap.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
