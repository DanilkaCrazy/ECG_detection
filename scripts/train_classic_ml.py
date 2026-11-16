import os
import json
from pathlib import Path
import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from ecgdet import config


def main(config_path="configs/classic_ml.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if cfg is None:
        cfg = {}
    # support flat or nested
    data_dir = Path(cfg.get('data_dir') or cfg.get('data', {}).get('data_dir') or 'data/processed/classic_ml')
    out_dir = Path(cfg.get('output_dir') or cfg.get('train', {}).get('output_dir') or 'reports/classic_ml')

    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    le = joblib.load(data_dir / 'label_encoder.pkl')

    models = {}
    models['lr'] = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)
    models['rf'] = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42)
    models['gb'] = GradientBoostingClassifier(n_estimators=200, random_state=42)

    results = {}
    for name, m in models.items():
        print(f'Training {name}...')
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        f1 = f1_score(y_val, pred, average='macro')
        acc = accuracy_score(y_val, pred)
        results[name] = {'f1_macro': float(f1), 'acc': float(acc), 'model': m}
        print(name, f1, acc)

    # pick best by f1
    best_name = max(results, key=lambda k: results[k]['f1_macro'])
    best = results[best_name]['model']
    print('Best:', best_name)
    pred_test = best.predict(X_test)
    print(classification_report(y_test, pred_test, target_names=le.classes_))
    print(confusion_matrix(y_test, pred_test))
    joblib.dump(best, out_dir / f'classic_ml_{best_name}.pkl')
    joblib.dump(le, out_dir / 'label_encoder.pkl')
    with open(out_dir / 'eval_test.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best': best_name,
            'f1_macro': float(f1_score(y_test, pred_test, average='macro')),
            'accuracy': float(accuracy_score(y_test, pred_test)),
            'report': classification_report(y_test, pred_test, target_names=le.classes_, output_dict=True)
        }, f, indent=2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/classic_ml.yaml')
    args = p.parse_args()
    main(args.config)
