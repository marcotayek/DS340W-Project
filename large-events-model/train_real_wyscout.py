"""Train and evaluate simple next-event baselines on converted public Wyscout CSVs.

NOT an implementation or exact reproduction of the original Large Events Model.
Uses only the previous event's six attributes as predictors.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

FEATURES = ['prev_h', 'prev_e', 'prev_x', 'prev_y', 'prev_t', 'prev_a']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='real_wyscout')
    args = parser.parse_args()
    folder = Path(args.data)
    if not folder.is_dir():
        raise SystemExit(f'Folder not found: {folder.resolve()}')
    results = {'dataset': 'Public Wyscout 2017/18, England', 'model': 'HistGradientBoosting previous-event baseline', 'metrics': {}}
    for target in ('e', 'x', 'y', 't'):
        train_path, val_path = folder / f'train_{target}.csv', folder / f'val_{target}.csv'
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        x_train, y_train = train[FEATURES], train['target']
        x_val, y_val = val[FEATURES], val['target']
        if target == 'e':
            model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, learning_rate=.1, random_state=42)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            scores = {'accuracy': float(accuracy_score(y_val, pred)), 'macro_f1': float(f1_score(y_val, pred, average='macro', zero_division=0))}
        else:
            model = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15, learning_rate=.1, random_state=42)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            scores = {'mae': float(mean_absolute_error(y_val, pred)), 'r2': float(r2_score(y_val, pred))}
            if target in ('x', 'y'):
                scores['mae_normalized_0_1'] = scores['mae'] / 100
            if target == 't':
                scores['mae_t_units'] = scores['mae']
        results['metrics'][target] = scores
        print(f'{target}: ' + ', '.join(f'{key}={value:.4f}' for key, value in scores.items()), flush=True)
    path = folder / 'baseline_metrics.json'
    path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nSaved metrics: {path.resolve()}')
    print('NOTE: This is a simplified baseline, NOT the original authors\' LEM or exact paper reproduction.')

if __name__ == '__main__':
    main()
