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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             log_loss, matthews_corrcoef, balanced_accuracy_score,
                             cohen_kappa_score)
from sklearn.model_selection import train_test_split

from ecgdet import normalize

ROOT = Path(r'D:\ИТМО\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\ECG_detection\ECG_detection')
PROCESSED = ROOT / 'data' / 'processed'
FIG = ROOT / 'reports' / 'figures' / 'classic_ml_full'
MET = ROOT / 'reports' / 'metrics' / 'classic_ml_full'
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

CML = PROCESSED / 'classic_ml'
X_train = np.load(CML / 'X_train.npy')
X_val = np.load(CML / 'X_val.npy')
X_test = np.load(CML / 'X_test.npy')
y_train = np.load(CML / 'y_train.npy')
y_val = np.load(CML / 'y_val.npy')
y_test = np.load(CML / 'y_test.npy')
le = joblib.load(CML / 'label_encoder.pkl')
classes = le.classes_.tolist()

X_train_s, _, y_train_s, _ = train_test_split(X_train, y_train, train_size=0.3, stratify=y_train, random_state=42)
X_val_s, _, y_val_s, _ = train_test_split(X_val, y_val, train_size=0.3, stratify=y_val, random_state=42)
X_test_s, _, y_test_s, _ = train_test_split(X_test, y_test, train_size=0.3, stratify=y_test, random_state=42)
print('sampled shapes:', X_train_s.shape, X_val_s.shape, X_test_s.shape)

# Drop columns that are all NaN
all_nan_cols = np.where(np.isnan(X_train_s).all(axis=0))[0]
if len(all_nan_cols) > 0:
    print(f'Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}')
    X_train_s = np.delete(X_train_s, all_nan_cols, axis=1)
    X_val_s = np.delete(X_val_s, all_nan_cols, axis=1)
    X_test_s = np.delete(X_test_s, all_nan_cols, axis=1)

# Fill remaining NaN values with median from train
train_median = np.nanmedian(X_train_s, axis=0)
for arr in [X_train_s, X_val_s, X_test_s]:
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.take(train_median, np.where(nan_mask)[1])

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

X_train_aug,y_train_aug = augment_feats(X_train_s,y_train_s,factor=0.3)

# Fill NaN in augmented data too
nan_mask = np.isnan(X_train_aug)
if nan_mask.any():
    X_train_aug[nan_mask] = np.take(train_median, np.where(nan_mask)[1])

norms = {}
norms['none'] = (X_train_s, X_val_s, X_test_s, X_train_aug)

cfg = normalize.RobustNormConfig(method='robust_z_mad')
fitted = normalize.fit_robust_norm(X_train_s,cfg)
norms['robust_median_mad'] = (normalize.apply_robust_norm(X_train_s,fitted),
                                normalize.apply_robust_norm(X_val_s,fitted),
                                normalize.apply_robust_norm(X_test_s,fitted),
                                normalize.apply_robust_norm(X_train_aug,fitted))

fitted_z = StandardScaler().fit(X_train_s)
norms['zscore'] = (fitted_z.transform(X_train_s), fitted_z.transform(X_val_s), fitted_z.transform(X_test_s), fitted_z.transform(X_train_aug))

fitted_m = MinMaxScaler().fit(X_train_s)
norms['minmax'] = (fitted_m.transform(X_train_s), fitted_m.transform(X_val_s), fitted_m.transform(X_test_s), fitted_m.transform(X_train_aug))

fitted_l2 = Normalizer(norm='l2').fit(X_train_s)
norms['l2'] = (fitted_l2.transform(X_train_s), fitted_l2.transform(X_val_s), fitted_l2.transform(X_test_s), fitted_l2.transform(X_train_aug))

models = {
    'RandomForest': RandomForestClassifier(n_estimators=150, class_weight='balanced', n_jobs=-1, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

results = []
detailed_results = []

for nname, (Xtrn, Xvn, Xtn, Xtrn_aug) in norms.items():
    print(f'\n=== Normalization: {nname} ===')
    for aug in [False, True]:
        Xtr = Xtrn_aug if aug else Xtrn
        ytr = y_train_aug if aug else y_train_s
        tag = f'{nname}_aug{int(aug)}'
        print(f'  Aug: {aug}')
        
        for mname, clf in models.items():
            try:
                print(f'    {mname}...', end=' ')
                clf.fit(Xtr, ytr)
                yp = clf.predict(Xtn)
                
                try:
                    yprob = clf.predict_proba(Xtn)
                    ll = log_loss(y_test_s, yprob, labels=list(range(len(classes))))
                except:
                    ll = np.nan
                
                acc = accuracy_score(y_test_s, yp)
                bal_acc = balanced_accuracy_score(y_test_s, yp)
                prec, rec, f1, sup = precision_recall_fscore_support(y_test_s, yp, average='macro')
                prec_w, rec_w, f1_w, sup_w = precision_recall_fscore_support(y_test_s, yp, average='weighted')
                mcc = matthews_corrcoef(y_test_s, yp)
                kappa = cohen_kappa_score(y_test_s, yp)
                rep = classification_report(y_test_s, yp, target_names=classes, output_dict=True)
                cm = confusion_matrix(y_test_s, yp)
                
                plt.figure(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.title(f'Confusion Matrix - {tag} - {mname}')
                plt.ylabel('True'); plt.xlabel('Predicted')
                plt.tight_layout()
                plt.savefig(FIG / f'cm_{tag}_{mname}.png', dpi=150)
                plt.close()
                
                res = {
                    'run': tag, 'norm': nname, 'aug': aug, 'model': mname,
                    'accuracy': acc, 'balanced_accuracy': bal_acc,
                    'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1,
                    'precision_weighted': prec_w, 'recall_weighted': rec_w, 'f1_weighted': f1_w,
                    'mcc': mcc, 'kappa': kappa, 'log_loss': ll
                }
                results.append(res)
                
                for c in classes:
                    if c in rep:
                        detailed_results.append({
                            'run': tag, 'norm': nname, 'aug': aug, 'model': mname, 'class': c,
                            'precision': rep[c]['precision'], 'recall': rep[c]['recall'],
                            'f1': rep[c]['f1-score'], 'support': rep[c]['support']
                        })
                
                (MET / f'report_{tag}_{mname}.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
                print(f'F1={f1:.4f}')
            except Exception as e:
                print(f'ERROR: {e}')

res_df = pd.DataFrame(results)
res_df.to_csv(MET / 'summary_metrics_all.csv', index=False)
det_df = pd.DataFrame(detailed_results)
det_df.to_csv(MET / 'per_class_metrics_all.csv', index=False)

for metric in ['f1_macro', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'mcc', 'kappa']:
    for mname in models.keys():
        sub = res_df[res_df['model']==mname]
        if sub.empty: continue
        pivot = sub.pivot(index='norm', columns='aug', values=metric)
        pivot.columns = ['no_aug', 'aug']
        plt.figure(figsize=(10,5))
        pivot.plot(kind='bar')
        plt.title(f'{mname} - {metric} by normalization & augmentation')
        plt.ylabel(metric); plt.xlabel('Normalization')
        plt.xticks(rotation=45); plt.legend(title='Augmentation')
        plt.tight_layout()
        plt.savefig(FIG / f'{mname}_{metric}_comparison.png', dpi=150)
        plt.close()

best_per_norm = res_df.groupby(['norm','aug']).agg({'f1_macro':'max'}).reset_index()
pivot = best_per_norm.pivot(index='norm', columns='aug', values='f1_macro')
pivot.columns = ['no_aug', 'aug']
plt.figure(figsize=(12,6))
pivot.plot(kind='bar')
plt.title('Best Model F1-macro by Normalization & Augmentation')
plt.ylabel('Best F1-macro'); plt.xlabel('Normalization')
plt.xticks(rotation=45); plt.legend(title='Augmentation')
plt.tight_layout()
plt.savefig(FIG / 'best_f1_macro_comparison.png', dpi=150)
plt.close()

lines = ['ECG Classic ML Full Evaluation Report', '='*50, f'Classes ({len(classes)}): {classes}', '']
for _, r in res_df.iterrows():
    lines.append(f"Run: {r['run']} | Model: {r['model']}")
    lines.append(f"  Acc: {r['accuracy']:.4f} | BalAcc: {r['balanced_accuracy']:.4f} | F1_macro: {r['f1_macro']:.4f} | F1_w: {r['f1_weighted']:.4f}")
    lines.append(f"  Prec_macro: {r['precision_macro']:.4f} | Rec_macro: {r['recall_macro']:.4f} | MCC: {r['mcc']:.4f} | Kappa: {r['kappa']:.4f} | LogLoss: {r['log_loss']:.4f}")
    lines.append('')

(MET / 'full_evaluation_report.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\nDone!')