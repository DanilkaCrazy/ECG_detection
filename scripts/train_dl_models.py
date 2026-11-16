import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             log_loss, matthews_corrcoef, balanced_accuracy_score,
                             cohen_kappa_score)

from ecgdet import normalize

ROOT = Path(r'D:\ИТМО\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\ECG_detection\ECG_detection')
PROCESSED = ROOT / 'data' / 'processed'
FIG = ROOT / 'reports' / 'figures' / 'dl_models'
MET = ROOT / 'reports' / 'metrics' / 'dl_models'
MODELS_DIR = ROOT / 'reports' / 'models' / 'dl_models'
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Load data
CML = PROCESSED / 'classic_ml'
X_train = np.load(CML / 'X_train.npy')
X_val = np.load(CML / 'X_val.npy')
X_test = np.load(CML / 'X_test.npy')
y_train = np.load(CML / 'y_train.npy')
y_val = np.load(CML / 'y_val.npy')
y_test = np.load(CML / 'y_test.npy')
le = joblib.load(CML / 'label_encoder.pkl')
classes = le.classes_.tolist()
num_classes = len(classes)

# Drop all-NaN columns
all_nan_cols = np.where(np.isnan(X_train).all(axis=0))[0]
if len(all_nan_cols) > 0:
    print(f'Dropping {len(all_nan_cols)} all-NaN columns')
    X_train = np.delete(X_train, all_nan_cols, axis=1)
    X_val = np.delete(X_val, all_nan_cols, axis=1)
    X_test = np.delete(X_test, all_nan_cols, axis=1)

# Fill remaining NaN with median
train_median = np.nanmedian(X_train, axis=0)
for arr in [X_train, X_val, X_test]:
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.take(train_median, np.where(nan_mask)[1])

print(f'Final feature dim: {X_train.shape[1]}')

# Normalizations
norms = {}
# none
norms['none'] = (X_train, X_val, X_test)

# robust median/MAD
cfg = normalize.RobustNormConfig(method='robust_z_mad')
fitted = normalize.fit_robust_norm(X_train, cfg)
norms['robust_median_mad'] = (normalize.apply_robust_norm(X_train, fitted),
                                normalize.apply_robust_norm(X_val, fitted),
                                normalize.apply_robust_norm(X_test, fitted))

# zscore
fitted_z = StandardScaler().fit(X_train)
norms['zscore'] = (fitted_z.transform(X_train), fitted_z.transform(X_val), fitted_z.transform(X_test))

# minmax
fitted_m = MinMaxScaler().fit(X_train)
norms['minmax'] = (fitted_m.transform(X_train), fitted_m.transform(X_val), fitted_m.transform(X_test))

# l2
fitted_l2 = Normalizer(norm='l2').fit(X_train)
norms['l2'] = (fitted_l2.transform(X_train), fitted_l2.transform(X_val), fitted_l2.transform(X_test))

# Class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Model definitions
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

class ResNet1D(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
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
        x = x.unsqueeze(1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

class BasicBlock1D(nn.Module):
    expansion = 1
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        # Treat each feature as a time step with input_size=1
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)
    
    def forward(self, x):
        # x: (B, L) -> (B, L, 1) treating each feature as a time step
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.dropout(out)
        return self.fc(out)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=10):
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
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
        
        # Validation
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if scheduler:
            scheduler.step(val_loss)
        
        if patience_counter >= patience:
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

results = []
detailed_results = []

for nname, (Xtr, Xvn, Xtn) in norms.items():
    print(f'\n=== Normalization: {nname} ===')
    
    # Create data loaders
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xvn_t = torch.tensor(Xvn, dtype=torch.float32)
    yvn_t = torch.tensor(y_val, dtype=torch.long)
    Xtn_t = torch.tensor(Xtn, dtype=torch.float32)
    ytn_t = torch.tensor(y_test, dtype=torch.long)
    
    # Weighted sampler for training
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_ds = TensorDataset(Xtr_t, ytr_t)
    val_ds = TensorDataset(Xvn_t, yvn_t)
    test_ds = TensorDataset(Xtn_t, ytn_t)
    
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Use only key normalizations for speed
    key_norms = ['none', 'robust_median_mad', 'zscore']
    norms = {k: v for k, v in norms.items() if k in key_norms}
    
    models_dict = {
        'CNN1D': CNN1D(Xtr.shape[1], num_classes).to(DEVICE),
        'MLP': MLPClassifier(Xtr.shape[1], num_classes).to(DEVICE),
    }
    
    for mname, model in models_dict.items():
        print(f'  Training {mname}...', end=' ')
        start = time.time()
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=6)
        
        # Evaluate
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
        
        # Save confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {nname} - {mname}')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(FIG / f'cm_{nname}_{mname}.png', dpi=150)
        plt.close()
        
        # Save training curves
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title(f'{mname} - {nname} Loss'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(history['val_f1'], label='Val F1')
        plt.title(f'{mname} - {nname} F1'); plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f'curves_{nname}_{mname}.png', dpi=150)
        plt.close()
        
        # Save model
        torch.save({
            'model_state': model.state_dict(),
            'model_class': mname,
            'input_dim': Xtr.shape[1],
            'num_classes': num_classes,
            'classes': classes,
            'normalization': nname,
            'history': history,
        }, MODELS_DIR / f'{mname}_{nname}.pt')
        
        res = {
            'run': f'{nname}_{mname}', 'norm': nname, 'model': mname,
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
                    'run': f'{nname}_{mname}', 'norm': nname, 'model': mname, 'class': c,
                    'precision': rep[c]['precision'], 'recall': rep[c]['recall'],
                    'f1': rep[c]['f1-score'], 'support': rep[c]['support']
                })
        
        (MET / f'report_{nname}_{mname}.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
        print(f'F1={f1:.4f} ({time.time()-start:.1f}s)')

res_df = pd.DataFrame(results)
res_df.to_csv(MET / 'summary_metrics_all.csv', index=False)
det_df = pd.DataFrame(detailed_results)
det_df.to_csv(MET / 'per_class_metrics_all.csv', index=False)

# Summary plots
for metric in ['f1_macro', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'mcc', 'kappa', 'log_loss']:
    for mname in models_dict.keys():
        sub = res_df[res_df['model']==mname]
        if sub.empty: continue
        plt.figure(figsize=(8,5))
        plt.bar(sub['norm'], sub[metric])
        plt.title(f'{mname} - {metric} by normalization')
        plt.ylabel(metric); plt.xlabel('Normalization')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIG / f'{mname}_{metric}_comparison.png', dpi=150)
        plt.close()

# Combined best comparison
best_per_norm = res_df.groupby('norm').agg({'f1_macro':'max'}).reset_index()
plt.figure(figsize=(10,6))
plt.bar(best_per_norm['norm'], best_per_norm['f1_macro'])
plt.title('Best Model F1-macro by Normalization')
plt.ylabel('Best F1-macro'); plt.xlabel('Normalization')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG / 'best_f1_macro_comparison.png', dpi=150)
plt.close()

# Text report
lines = ['ECG Deep Learning Evaluation Report', '='*50, f'Classes ({num_classes}): {classes}', '']
for _, r in res_df.iterrows():
    lines.append(f"Run: {r['run']}")
    lines.append(f"  Acc: {r['accuracy']:.4f} | BalAcc: {r['balanced_accuracy']:.4f} | F1_macro: {r['f1_macro']:.4f} | F1_w: {r['f1_weighted']:.4f}")
    lines.append(f"  Prec_macro: {r['precision_macro']:.4f} | Rec_macro: {r['recall_macro']:.4f} | MCC: {r['mcc']:.4f} | Kappa: {r['kappa']:.4f} | LogLoss: {r['log_loss']:.4f} | Time: {r['train_time']:.1f}s")
    lines.append('')

(MET / 'full_evaluation_report.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\nDone!')