import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(r'D:\ИТМО\Разработка нейросетевого алгоритма детекции и классификации нарушений ритма сердца\ECG_detection\ECG_detection')
FIG = ROOT / 'reports' / 'figures' / 'analysis'
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

# Load unified manifest
df = pd.read_csv(ROOT / 'data' / 'processed' / 'unified_ecg_manifest.csv')

# 1. Overall class distribution
plt.figure(figsize=(12, 6))
class_counts = df['rhythm'].value_counts().sort_values(ascending=False)
ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
ax.set_title('Распределение записей по классам ритмов (Unified Manifest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Класс ритма', fontsize=12)
ax.set_ylabel('Количество записей', fontsize=12)
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'class_distribution_overall.png', dpi=200)
plt.close()

# 2. Class distribution by source
plt.figure(figsize=(12, 6))
cross = df.groupby(['source', 'rhythm']).size().unstack(fill_value=0)
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
ax = cross.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
ax.set_title('Распределение классов по источникам данных', fontsize=14, fontweight='bold')
ax.set_xlabel('Источник', fontsize=12)
ax.set_ylabel('Количество записей', fontsize=12)
ax.legend(title='Класс ритма', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG / 'class_distribution_by_source.png', dpi=200)
plt.close()

# 3. Percentage by source
plt.figure(figsize=(12, 6))
ax = cross_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
ax.set_title('Процентное распределение классов по источникам', fontsize=14, fontweight='bold')
ax.set_xlabel('Источник', fontsize=12)
ax.set_ylabel('Процент (%)', fontsize=12)
ax.legend(title='Класс ритма', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG / 'class_distribution_by_source_pct.png', dpi=200)
plt.close()

# 4. Patient-level distribution (if available)
if 'patient_id' in df.columns and df['patient_id'].notna().any():
    patient_class = df.dropna(subset=['patient_id']).groupby('patient_id')['rhythm'].first()
    if len(patient_class) > 0:
        plt.figure(figsize=(12, 6))
        pc = patient_class.value_counts().sort_values(ascending=False)
        ax = sns.barplot(x=pc.index, y=pc.values, palette='viridis')
        ax.set_title('Распределение пациентов по классам ритмов', fontsize=14, fontweight='bold')
        ax.set_xlabel('Класс ритма', fontsize=12)
        ax.set_ylabel('Количество пациентов', fontsize=12)
        for i, v in enumerate(pc.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIG / 'class_distribution_patients.png', dpi=200)
        plt.close()

# 5. Class imbalance ratio
plt.figure(figsize=(10, 5))
imbalance = class_counts / class_counts.max()
ax = sns.barplot(x=imbalance.index, y=imbalance.values, palette='Reds_r')
ax.set_title('Дисбаланс классов (относительно максимального класса)', fontsize=14, fontweight='bold')
ax.set_xlabel('Класс ритма', fontsize=12)
ax.set_ylabel('Отношение к мажоритарному классу', fontsize=12)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
for i, v in enumerate(imbalance.values):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(FIG / 'class_imbalance_ratio.png', dpi=200)
plt.close()

# Print summary
print('=== Class Distribution Summary ===')
print(f'Total records: {len(df):,}')
print(f'Total classes: {df["rhythm"].nunique()}')
print(f'Sources: {df["source"].nunique()}')
print()
for c in class_counts.index:
    cnt = class_counts[c]
    pct = cnt / len(df) * 100
    print(f'  {c}: {cnt:,} ({pct:.1f}%)')
print()
print('By source:')
print(df.groupby('source')['rhythm'].value_counts().unstack(fill_value=0).to_string())