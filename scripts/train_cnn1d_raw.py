import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time
import sys

sys.path.insert(0, r'D:\ИТМО\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\ECG_detection\ECG_detection\src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             log_loss, matthews_corrcoef, balanced_accuracy_score,
                             cohen_kappa_score)

from ecgdet import config

ROOT = Path(r'D:\ИТМО\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\ECG_detection\ECG_detection')
PROCESSED = ROOT / 'data' / 'processed'
FIG = ROOT / 'reports' / 'figures' / 'cnn1d_raw'
MET = ROOT / 'reports' / 'metrics' / 'cnn1d_raw'
MODELS_DIR = ROOT / 'reports' / 'models' / 'cnn1d_raw'
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

DEVICE = torch.device('cpu')
print(f'Using device: {DEVICE}', flush=True)

# Load manifest
df = pd.read_csv(PROCESSED / 'unified_ecg_manifest.csv')

# Use stratified sample
MAX_PER_CLASS = 300
sampled = []
for cls in df['rhythm'].unique():
    sub = df[df['rhythm'] == cls]
    n = min(len(sub), MAX_PER_CLASS)
    sampled.append(sub.sample(n=n, random_state=42))
df_sampled = pd.concat(sampled).reset_index(drop=True)
print(f'Sampled {len(df_sampled)} records', flush=True)

TARGET_FS = 500
TARGET_LEN = 2500  # 5 seconds at 500 Hz
N_LEADS = 12

def load_raw_signal(row):
    try:
        rec = wfdb.rdrecord(row['path_hea'].replace('.hea', ''))
        sig = rec.p_signal.astype(np.float32)
        fs = rec.fs
        if abs(fs - TARGET_FS) > 1e-6:
            from math import gcd
            g = gcd(TARGET_FS, int(fs))
            up = TARGET_FS // g
            down = int(fs) // g
            sig = resample_poly(sig, up, down, axis=0).astype(np.float32)
        if sig.shape[0] > TARGET_LEN:
            sig = sig[:TARGET_LEN]
        elif sig.shape[0] < TARGET_LEN:
            pad = TARGET_LEN - sig.shape[0]
            sig = np.pad(sig, ((0, pad), (0, 0)), mode='constant')
        if sig.shape[1] < N_LEADS:
            pad_leads = N_LEADS - sig.shape[1]
            sig = np.pad(sig, ((0, 0), (0, pad_leads)), mode='constant')
        elif sig.shape[1] > N_LEADS:
            sig = sig[:, :N_LEADS]
        sig = np.nan_to_num(sig, nan=0.0)
        return sig
    except Exception as e:
        return None

print('Loading signals...', flush=True)
X_list = []
y_list = []
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['rhythm'].unique())
classes = le.classes_.tolist()
num_classes = len(classes)

for idx, row in df_sampled.iterrows():
    sig = load_raw_signal(row)
    if sig is not None:
        X_list.append(sig)
        y_list.append(row['rhythm'])
    if (idx + 1) % 500 == 0:
        print(f'  Loaded {idx+1}/{len(df_sampled)}', flush=True)

X = np.array(X_list, dtype=np.float32)
y = le.transform(y_list)
print(f'X shape: {X.shape}, y shape: {y.shape}', flush=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

# Per-lead standardization
mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1)) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Transpose to (N, 12, 2500)
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}', flush=True)

np.save(MET / 'norm_mean.npy', mean)
np.save(MET / 'norm_std.npy', std)
joblib.dump(le, MET / 'label_encoder.pkl')

# Class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-8)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Model: Simple 1D CNN
class CNN1D(nn.Module):
    def __init__(self, num_classes, in_channels=12, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

class ResNet1D(nn.Module):
    def __init__(self, num_classes, in_channels=12, dropout=0.3):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(BasicBlock1D(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            else:
                layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=15, patience=5):
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        val_acc = accuracy_score(all_targets, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if scheduler:
            scheduler.step(val_loss)
        
        print(f'    Epoch {epoch+1}: loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}', flush=True)
        
        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(best_state)
    return model, history

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

from sklearn.metrics import f1_score

# Augmentation
def augment_signal(sig):
    sig_aug = sig.copy()
    sig_aug += np.random.normal(0, 0.05, sig_aug.shape).astype(np.float32)
    scale = np.random.uniform(0.95, 1.05)
    sig_aug *= scale
    return sig_aug

def augment_dataset(X, y, factor=0.5):
    n = int(len(X) * factor)
    idx = np.random.choice(len(X), n, replace=True)
    X_aug = np.array([augment_signal(X[i]) for i in idx], dtype=np.float32)
    y_aug = y[idx]
    return np.concatenate([X, X_aug], axis=0), np.concatenate([y, y_aug], axis=0)

def create_loaders(Xtr, ytr, Xvn, yvn, Xtn, ytn, batch_size=32):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xvn_t = torch.tensor(Xvn, dtype=torch.float32)
    yvn_t = torch.tensor(yvn, dtype=torch.long)
    Xtn_t = torch.tensor(Xtn, dtype=torch.float32)
    ytn_t = torch.tensor(ytn, dtype=torch.long)
    
    sample_weights = class_weights[ytr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_ds = TensorDataset(Xtr_t, ytr_t)
    val_ds = TensorDataset(Xvn_t, yvn_t)
    test_ds = TensorDataset(Xtn_t, ytn_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader

results = []
detailed_results = []

for aug in [False, True]:
    if aug:
        Xtr, ytr = augment_dataset(X_train, y_train, factor=0.5)
        tag = 'aug1'
    else:
        Xtr, ytr = X_train.copy(), y_train.copy()
        tag = 'aug0'
    
    print(f'\n=== Augmentation: {aug} ===', flush=True)
    
    class_counts = np.bincount(ytr)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    train_loader, val_loader, test_loader = create_loaders(Xtr, ytr, X_val, y_val, X_test, y_test)
    
    models_dict = {
        'CNN1D': CNN1D(num_classes, in_channels=12).to(DEVICE),
        'ResNet1D': ResNet1D(num_classes, in_channels=12).to(DEVICE),
    }
    
    for mname, model in models_dict.items():
        print(f'  Training {mname}...', flush=True)
        start = time.time()
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=15, patience=5)
        
        y_true, y_pred, y_prob = evaluate_model(model, test_loader)
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        ll = log_loss(y_true, y_prob, labels=list(range(num_classes)))
        rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {mname} - aug{int(aug)}')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(FIG / f'cm_{mname}_aug{int(aug)}.png', dpi=150)
        plt.close()
        
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title(f'{mname} - Loss'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(history['val_f1'], label='Val F1')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title(f'{mname} - F1/Acc'); plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f'curves_{mname}_aug{int(aug)}.png', dpi=150)
        plt.close()
        
        torch.save({
            'model_state': model.state_dict(),
            'model_class': mname,
            'num_classes': num_classes,
            'classes': classes,
            'mean': mean,
            'std': std,
            'augmentation': aug,
            'history': history,
        }, MODELS_DIR / f'{mname}_aug{int(aug)}.pt')
        
        res = {
            'run': f'{mname}_aug{int(aug)}', 'model': mname, 'aug': aug,
            'accuracy': acc, 'balanced_accuracy': bal_acc,
            'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1,
            'precision_weighted': prec_w, 'recall_weighted': rec_w, 'f1_weighted': f1_w,
            'mcc': mcc, 'kappa': kappa, 'log_loss': ll,
            'train_time': time.time() - start
        }
        results.append(res)
        
        for c in classes:
            if c in rep:
                detailed_results.append({
                    'run': f'{mname}_aug{int(aug)}', 'model': mname, 'aug': aug, 'class': c,
                    'precision': rep[c]['precision'], 'recall': rep[c]['recall'],
                    'f1': rep[c]['f1-score'], 'support': rep[c]['support']
                })
        
        (MET / f'report_{mname}_aug{int(aug)}.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
        print(f'  {mname}: F1={f1:.4f} Acc={acc:.4f} ({time.time()-start:.1f}s)', flush=True)

res_df = pd.DataFrame(results)
res_df.to_csv(MET / 'summary_metrics_all.csv', index=False)
det_df = pd.DataFrame(detailed_results)
det_df.to_csv(MET / 'per_class_metrics_all.csv', index=False)

# Summary plot
plt.figure(figsize=(10,6))
for mname in models_dict.keys():
    sub = res_df[res_df['model']==mname]
    x = [f'{mname}_aug{int(a)}' for a in sub['aug']]
    plt.bar(x, sub['f1_macro'], label=mname)
plt.title('1D-CNN on Raw ECG - F1-macro by Model & Augmentation')
plt.ylabel('F1-macro'); plt.xlabel('Run')
plt.xticks(rotation=45); plt.legend()
plt.tight_layout()
plt.savefig(FIG / 'f1_macro_comparison.png', dpi=150)
plt.close()

# Text report
lines = ['ECG 1D-CNN on Raw Signals Evaluation Report', '='*50, f'Classes ({num_classes}): {classes}', f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}', '']
for _, r in res_df.iterrows():
    lines.append(f"Run: {r['run']}")
    lines.append(f"  Acc: {r['accuracy']:.4f} | BalAcc: {r['balanced_accuracy']:.4f} | F1_macro: {r['f1_macro']:.4f} | F1_w: {r['f1_weighted']:.4f}")
    lines.append(f"  Prec_macro: {r['precision_macro']:.4f} | Rec_macro: {r['recall_macro']:.4f} | MCC: {r['mcc']:.4f} | Kappa: {r['kappa']:.4f} | LogLoss: {r['log_loss']:.4f} | Time: {r['train_time']:.1f}s")
    lines.append('')

(MET / 'full_evaluation_report.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\nDone!', flush=True)