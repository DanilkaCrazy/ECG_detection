import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             log_loss, matthews_corrcoef)

from ecgdet import normalize

ROOT = Path(r'D:\ИТМО\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\ECG_detection\ECG_detection')
PROCESSED = ROOT / 'data' / 'processed'
FIG = ROOT / 'reports' / 'figures' / 'analysis'
MET = ROOT / 'reports' / 'metrics'
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

# 1) Классы
unif = pd.read_csv(PROCESSED / 'unified_ecg_manifest.csv')
class_counts = unif['rhythm'].value_counts()
plt.figure(figsize=(10,6))
ax = sns.barplot(x=class_counts.index, y=class_counts.values)
ax.set_title('Распределение по классам ритмов (unified_ecg_manifest)')
ax.set_xlabel('Класс'); ax.set_ylabel('Количество')
for i,v in enumerate(class_counts.values): ax.text(i,v+100,str(v),ha='center')
plt.tight_layout(); plt.savefig(FIG/'01_class_distribution_unified.png',dpi=150); plt.close()

# 2) Данные
CML = PROCESSED / 'classic_ml'
X_train = np.load(CML/'X_train.npy'); X_val = np.load(CML/'X_val.npy'); X_test = np.load(CML/'X_test.npy')
y_train = np.load(CML/'y_train.npy'); y_val = np.load(CML/'y_val.npy'); y_test = np.load(CML/'y_test.npy')
le = joblib.load(CML/'label_encoder.pkl')
classes = le.classes_.tolist()

# 3) Аугментация
rng = np.random.RandomState(42)
def augment_feats(X,y,factor=0.3,noise_std=0.005):
    if factor<=0: return X.copy(),y.copy()
    n=int(len(X)*factor)
    idx=rng.choice(len(X),n,replace=True)
    Xa=X[idx].copy()
    std=np.std(Xa,axis=0,keepdims=True)+1e-12
    Xa += rng.normal(0,noise_std,Xa.shape)*std
    Xa *= rng.uniform(0.99,1.01,(n,1))
    return np.concatenate([X,Xa],axis=0), np.concatenate([y,y[idx]],axis=0)
X_train_aug,y_train_aug = augment_feats(X_train,y_train,factor=0.3)

# 4) Нормализации
cfg = normalize.RobustNormConfig(method='robust_z_mad')
fitted = normalize.fit_robust_norm(X_train,cfg)
Xtr_r = normalize.apply_robust_norm(X_train,fitted); Xv_r = normalize.apply_robust_norm(X_val,fitted); Xt_r = normalize.apply_robust_norm(X_test,fitted)
Xtr_r_aug = normalize.apply_robust_norm(X_train_aug,fitted)

fitted_z = StandardScaler().fit(X_train)
Xtr_z = fitted_z.transform(X_train); Xv_z = fitted_z.transform(X_val); Xt_z = fitted_z.transform(X_test)
Xtr_z_aug = fitted_z.transform(X_train_aug)

fitted_m = MinMaxScaler().fit(X_train)
Xtr_m = fitted_m.transform(X_train); Xv_m = fitted_m.transform(X_val); Xt_m = fitted_m.transform(X_test)
Xtr_m_aug = fitted_m.transform(X_train_aug)

fitted_l2 = Normalizer(norm='l2').fit(X_train)
Xtr_l2 = fitted_l2.transform(X_train); Xv_l2 = fitted_l2.transform(X_val); Xt_l2 = fitted_l2.transform(X_test)
Xtr_l2_aug = fitted_l2.transform(X_train_aug)

norms = {
 'none': (X_train,X_val,X_test,X_train_aug),
 'robust_median_mad': (Xtr_r,Xv_r,Xt_r,Xtr_r_aug),
 'zscore': (Xtr_z,Xv_z,Xt_z,Xtr_z_aug),
 'minmax': (Xtr_m,Xv_m,Xt_m,Xtr_m_aug),
 'l2': (Xtr_l2,Xv_l2,Xt_l2,Xtr_l2_aug),
}

# 5) Сравнение
results=[]
for nname,(Xtrn,Xvn,Xtn,Xtrn_aug) in norms.items():
    for aug in [False,True]:
        Xtr=Xtrn_aug if aug else Xtrn; ytr=y_train_aug if aug else y_train
        clf=RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=-1,random_state=123)
        clf.fit(Xtr,ytr)
        yp=clf.predict(Xtn)
        try:
            yprob=clf.predict_proba(Xtn)
            ll=log_loss(y_test,yprob,labels=list(range(len(classes))))
        except: ll=float('nan')
        acc=accuracy_score(y_test,yp)
        prec,rec,f1,sup=precision_recall_fscore_support(y_test,yp,average='macro')
        prec_w,rec_w,f1_w,sup_w=precision_recall_fscore_support(y_test,yp,average='weighted')
        mcc=matthews_corrcoef(y_test,yp)
        rep=classification_report(y_test,yp,target_names=classes,output_dict=True)
        cm=confusion_matrix(y_test,yp)
        tag=f'{nname}_aug{int(aug)}'
        plt.figure(figsize=(7,5))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes)
        plt.title(f'Confusion Matrix - {tag}'); plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout(); plt.savefig(FIG/f'cm_{tag}.png',dpi=150); plt.close()
        results.append({'run':tag,'norm':nname,'aug':aug,'accuracy':acc,'precision_macro':prec,'recall_macro':rec,'f1_macro':f1,'precision_weighted':prec_w,'recall_weighted':rec_w,'f1_weighted':f1_w,'log_loss':ll,'mcc':mcc})
        (MET/f'report_{tag}.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')

res_df=pd.DataFrame(results); res_df.to_csv(MET/'summary_metrics_all.csv',index=False)

# графики
plt.figure(figsize=(14,6))
pivot=res_df.pivot(index='norm',columns='aug',values='f1_macro'); pivot.columns=['no_aug','aug']
pivot.plot(kind='bar'); plt.title('Сравнение F1-macro по нормализациям и аугментации'); plt.ylabel('F1-macro'); plt.xticks(rotation=45); plt.legend(title='Аугментация')
plt.tight_layout(); plt.savefig(FIG/'02_f1_macro_comparison.png',dpi=150); plt.close()

plt.figure(figsize=(14,6))
pivot2=res_df.pivot(index='norm',columns='aug',values='accuracy'); pivot2.columns=['no_aug','aug']
pivot2.plot(kind='bar'); plt.title('Сравнение Accuracy по нормализациям и аугментации'); plt.ylabel('Accuracy'); plt.xticks(rotation=45); plt.legend(title='Аугментация')
plt.tight_layout(); plt.savefig(FIG/'03_accuracy_comparison.png',dpi=150); plt.close()

# отчёт
lines=['ECG data analysis report','===================================',f'Classes ({len(classes)}): {classes}',f'Unified total: {len(unif)}','Class counts (unified):']
for k,v in class_counts.items(): lines.append(f'  {k}: {v}')
lines.append(''); lines.append('Metrics (test) by run:')
for _,r in res_df.iterrows():
    lines.append(f"Run: {r['run']} | norm={r['norm']} aug={r['aug']}")
    lines.append(f"  Accuracy: {r['accuracy']:.4f} | Precision_macro: {r['precision_macro']:.4f} | Recall_macro: {r['recall_macro']:.4f} | F1_macro: {r['f1_macro']:.4f}")
    lines.append(f"  Precision_weighted: {r['precision_weighted']:.4f} | Recall_weighted: {r['recall_weighted']:.4f} | F1_weighted: {r['f1_weighted']:.4f} | MCC: {r['mcc']:.4f} | LogLoss: {r['log_loss']:.4f}")
    lines.append('')
(MET/'analysis_report.txt').write_text('\n'.join(lines),encoding='utf-8')
print('done')
