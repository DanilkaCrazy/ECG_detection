"""Stage 3.5 P2 — compare advanced atrial features on the SAME fixed split.

Feature sets: expert, expert+v2_atrial (Stage3.4 baseline), advanced_atrial,
expert+advanced_atrial, v2+advanced_atrial, expert+v2+advanced_atrial.
Models: HistGB, CatBoost, XGBoost, LightGBM, SVC, RandomForest.
Selection on independent `sel` split; thresholds from `thr` split only.
Tests on the same `te` indices as Stage 3.4 P0 baseline.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from ecgdet import config

ROOT = Path(__file__).resolve().parents[1]
D = config.PROCESSED_ROOT / "af_afl_stage34"
MET = ROOT / "reports" / "metrics" / "af_afl_stage35"
FIG = ROOT / "reports" / "figures" / "af_afl_stage35"
MOD = ROOT / "reports" / "models" / "af_afl_stage35"
for p in (MET, FIG, MOD): p.mkdir(parents=True, exist_ok=True)
THR = np.linspace(0.05, 0.95, 91)
SETS = {
 "expert": ["expert"],
 "stage34_baseline": ["expert","v2","atrial","heat"],
 "atrial_advanced": ["atrial_adv"],
 "expert_atrial_adv": ["expert","atrial_adv"],
 "v2_atrial_adv": ["v2","atrial_adv"],
 "all_atrial_adv": ["expert","v2","atrial_adv","heat"],
}
FILES = {"expert":"features_advanced_expert.npy","v2":"features_advanced_v2.npy",
         "atrial":"features_atrial34.npy","atrial_adv":"features_atrial_advanced.npy",
         "heat":"features_heat.npy"}
MODELS = {
 "HistGB": lambda p,n: HistGradientBoostingClassifier(max_iter=250,learning_rate=.06,max_leaf_nodes=20,min_samples_leaf=25,l2_regularization=.5,class_weight="balanced",random_state=42),
 "CatBoost": lambda p,n: CatBoostClassifier(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3,loss_function="Logloss",eval_metric="AUC",auto_class_weights="Balanced",verbose=False,thread_count=8,random_seed=42),
 "XGBoost": lambda p,n: XGBClassifier(n_estimators=400,max_depth=4,learning_rate=.05,min_child_weight=15,subsample=.9,colsample_bytree=.85,scale_pos_weight=n/max(p,1),tree_method="hist",eval_metric="auc",n_jobs=8,random_state=42),
 "LightGBM": lambda p,n: LGBMClassifier(n_estimators=400,num_leaves=31,learning_rate=.05,min_child_samples=25,class_weight="balanced",n_jobs=8,verbose=-1,random_state=42),
 "SVC": lambda p,n: SVC(C=2.,kernel="rbf",probability=True,class_weight="balanced",random_state=42),
 "RandomForest": lambda p,n: RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight="balanced_subsample",n_jobs=-1,random_state=42),
}
def evaluate(y,p,t):
    q=(p>=t).astype(int)
    return dict(f1_macro=f1_score(y,q,average="macro"),afib_f1=f1_score(y,q,pos_label=1),
                afl_f1=f1_score(y,q,pos_label=0),afib_recall=recall_score(y,q,pos_label=1),
                afib_precision=precision_score(y,q,pos_label=1),accuracy=accuracy_score(y,q),
                balanced_accuracy=balanced_accuracy_score(y,q),mcc=matthews_corrcoef(y,q),
                pr_auc_afib=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),
                log_loss=log_loss(y,np.clip(p,1e-12,1-1e-12)),brier=brier_score_loss(y,p))
def main():
    man=pd.read_csv(D/"meta.csv"); y=man.label.to_numpy(); groups=man.patient_key.to_numpy()
    sp=pd.read_csv(D/"split.csv"); tr0,held=np.where(sp.split=="train")[0],np.where(sp.split!="train")[0]
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=42).split(held,y[held],groups=groups[held]))
    cal,thr_i,sel,te=[held[folds[i][1]] for i in range(4)]
    print(f"same split as baseline: train={len(tr0)} cal={len(cal)} thr={len(thr_i)} sel={len(sel)} test={len(te)}",flush=True)
    rows=[]; cache={}
    for name,parts in SETS.items():
        if name not in cache:
            X=np.concatenate([np.load(D/FILES[p],mmap_mode="r") for p in parts],axis=1)
            an=np.isnan(X[tr0]).all(axis=0); X=np.delete(X,np.where(an)[0],axis=1)
            med=np.nan_to_num(np.nanmedian(X[tr0],axis=0)); X=np.where(np.isnan(X),med,X)
            q1,q3=np.percentile(X[tr0],[25,75],axis=0); sc=np.where((q3-q1)>1e-9,q3-q1,1.0)
            cen=np.median(X[tr0],axis=0); X=np.clip((X-cen)/sc,-20,20).astype(np.float32)
            cache[name]=X
        X=cache[name]
        for mn,factory in MODELS.items():
            pos,neg=float((y[tr0]==1).sum()),float((y[tr0]==0).sum())
            m=factory(pos,neg); m.fit(X[tr0],y[tr0])
            cal_m=LogisticRegression().fit(m.predict_proba(X[cal])[:,1].reshape(-1,1),y[cal])
            pthr=cal_m.predict_proba(m.predict_proba(X[thr_i])[:,1].reshape(-1,1))[:,1]
            thr=float(THR[int(np.argmax([f1_score(y[thr_i],(pthr>=t).astype(int),average="macro") for t in THR]))])
            psel=cal_m.predict_proba(m.predict_proba(X[sel])[:,1].reshape(-1,1))[:,1]
            sf=f1_score(y[sel],(psel>=thr).astype(int),average="macro")
            pte=cal_m.predict_proba(m.predict_proba(X[te])[:,1].reshape(-1,1))[:,1]
            r=evaluate(y[te],pte,thr); r.update({"features":name,"model":mn,"select_f1":sf,"threshold":thr,"n_test":len(te)})
            rows.append(r)
            joblib.dump({"model":m,"calibrator":cal_m,"threshold":thr,"features":name},MOD/f"{name}_{mn}.joblib")
        print(name,"done",flush=True)
    df=pd.DataFrame(rows); df.to_csv(MET/"stage35_results.csv",index=False)
    summ=[]
    for n in SETS:
        s=df[df.features==n].sort_values("select_f1",ascending=False)
        if len(s):
            b=s.iloc[0]; summ.append(f"- {n}: best={b['model']} selectF1={b['select_f1']:.4f} testF1={b['f1_macro']:.4f} AFIB-F1={b['afib_f1']:.4f} AFIB-recall={b['afib_recall']:.4f}")
    (ROOT/"reports"/"AF_AFL_STAGE35_BASELINE.md").write_text(
        "# Stage 3.5 — сравнение признаков на одном split\n\n"+"\n".join(summ)+"\n\n"+df.sort_values("select_f1",ascending=False).head(25).to_string(index=False),encoding="utf-8")
    print("DONE",flush=True)
if __name__=="__main__": raise SystemExit(main())
