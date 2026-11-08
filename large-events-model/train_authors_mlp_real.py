"""Real Wyscout next-event experiment using the MLP class from the authors' training script.

Adaptation, NOT an exact reproduction of Mendes-Neves et al. (different public
Wyscout dataset, features, targets, train/validation split and training protocol).
"""
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ['prev_h', 'prev_e', 'prev_x', 'prev_y', 'prev_t', 'prev_a']
TASKS = ['e', 'x', 'y', 't']


def load_mlp(path):
    spec = importlib.util.spec_from_file_location('authors_tabular_lem', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='real_wyscout')
    ap.add_argument('--authors-script', default='0001 Train Tabular LEMs.py')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--hidden', type=int, default=196)
    ap.add_argument('--learning-rate', type=float, default=0.001)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1: ap.error('epochs and batch-size must be positive')
    folder = Path(args.data)
    source = Path(args.authors_script)
    if not source.exists():
        raise SystemExit(f'Cannot find authors training script: {source.resolve()}')
    MLP = load_mlp(source)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}; MLP imported from: {source.resolve()}', flush=True)
    output = {'dataset': 'Public Wyscout England 2017/18 (80 matches)',
              'model': 'Original authors MLP class; adapted six-feature previous-event task',
              'exact_reproduction': False, 'epochs': args.epochs, 'hidden_sizes': [args.hidden]*3,
              'metrics': {}}
    for task in TASKS:
        tr = pd.read_csv(folder / f'train_{task}.csv')
        va = pd.read_csv(folder / f'val_{task}.csv')
        # Scale features using training-only constants, avoiding validation leakage.
        scales = np.array([1., 31., 100., 100., 100., 1.], dtype=np.float32)
        xtr = (tr[FEATURES].to_numpy(dtype=np.float32) / scales).copy()
        xva = (va[FEATURES].to_numpy(dtype=np.float32) / scales).copy()
        if task == 'e':
            classes = np.sort(tr['target'].unique())
            label_map = {int(v): i for i, v in enumerate(classes)}
            if not set(va['target'].unique()).issubset(set(classes)):
                raise SystemExit('Validation contains event classes absent from training.')
            ytr = np.array([label_map[int(v)] for v in tr['target']], dtype=np.int64)
            yva = va['target'].to_numpy(dtype=np.int64)
            output_size = len(classes)
            loss_fn = nn.CrossEntropyLoss()
            y_tensor = torch.from_numpy(ytr)
        else:
            # Keep regression output in original units for direct baseline comparison.
            ytr = tr['target'].to_numpy(dtype=np.float32)
            yva = va['target'].to_numpy(dtype=np.float32)
            # Predict normalized targets to improve optimization stability.
            target_scale = 100.0 if task in ('x', 'y') else 10.0
            y_tensor = torch.from_numpy((ytr / target_scale).reshape(-1, 1))
            output_size = 1
            loss_fn = nn.MSELoss()
        model = MLP(len(FEATURES), [args.hidden]*3, output_size, dropout_rate=0.3).to(device)
        loader = DataLoader(TensorDataset(torch.from_numpy(xtr), y_tensor),
                            batch_size=args.batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):
            model.train(); total = 0.0; seen = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward(); opt.step()
                total += loss.item() * len(xb); seen += len(xb)
            print(f'{task} epoch {epoch+1}/{args.epochs}: train_loss={total/seen:.5f}', flush=True)
        model.eval(); parts = []
        with torch.no_grad():
            for i in range(0, len(xva), args.batch_size):
                pred = model(torch.from_numpy(xva[i:i+args.batch_size]).to(device))
                parts.append(pred.cpu().numpy())
        preds = np.concatenate(parts, axis=0)
        if task == 'e':
            predicted = classes[np.argmax(preds, axis=1)]
            metrics = {'accuracy': float(accuracy_score(yva, predicted)),
                       'macro_f1': float(f1_score(yva, predicted, average='macro', zero_division=0))}
        else:
            predicted = preds[:, 0] * target_scale
            metrics = {'mae': float(mean_absolute_error(yva, predicted)),
                       'r2': float(r2_score(yva, predicted))}
            if task in ('x', 'y'):
                metrics['mae_normalized_0_1'] = metrics['mae'] / 100.0
        output['metrics'][task] = metrics
        print(f'{task}: ' + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()), flush=True)
        torch.save({'state_dict': model.state_dict(), 'features': FEATURES,
                    'feature_scales': scales.tolist(), 'hidden_sizes': [args.hidden]*3,
                    'classes': classes.tolist() if task == 'e' else None,
                    'target_scale': None if task == 'e' else target_scale},
                   folder / f'authors_mlp_{task}.pt')
    path = folder / 'authors_mlp_metrics.json'
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'\nSaved metrics: {path.resolve()}')
    print('NOTE: Original authors MLP architecture, adapted task; NOT an exact paper reproduction.')

if __name__ == '__main__':
    main()
