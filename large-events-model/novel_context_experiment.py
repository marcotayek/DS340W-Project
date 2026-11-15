"""Compare previous-event context windows using the authors' MLP architecture.

This is a NEW experiment on adapted public Wyscout data, not a reproduction of
the original paper's benchmark. Only TRAIN and TEST are used by default; the
held-out VAL split requires an explicit final-evaluation flag.

Example (PowerShell, from this script's directory):
    py -3.12 novel_context_experiment.py --data real_wyscout_70_20_10 --epochs 30
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


ATTRIBUTES = ("h", "e", "x", "y", "t", "a")
SCALES = {"h": 1.0, "e": 31.0, "x": 100.0, "y": 100.0, "t": 100.0, "a": 1.0}
VARIANTS = (("previous_1", 1, False),
            ("previous_3", 3, False),
            ("previous_5", 5, False),
            ("previous_5_transitions", 5, True))


def build_event_history(raw, max_history=5):
    """Build lagged event features without letting history cross match boundaries.

    All variants use rows with five available previous events. This common
    sample keeps comparisons fair (the one-event baseline gets the same rows).
    The CSV must be sorted chronologically *within each match*, as written by
    convert_public_wyscout.py. Its row order is deliberately preserved.
    """
    required = {"match_id", *ATTRIBUTES}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw event CSV missing columns: {sorted(missing)}")
    if max_history < 1:
        raise ValueError("max_history must be positive")
    if raw[list(required)].isnull().any().any():
        raise ValueError("Raw events contain null match IDs or event attributes")

    frame = raw.copy()
    grouped = frame.groupby("match_id", sort=False)
    for lag in range(1, max_history + 1):
        for attr in ATTRIBUTES:
            frame[f"lag{lag}_{attr}"] = grouped[attr].shift(lag)
    # Only history columns are used to establish eligibility; never drop rows
    # due to a target value. The prediction target is the CURRENT event.
    frame = frame.dropna(subset=[f"lag{max_history}_{a}" for a in ATTRIBUTES]).copy()
    return frame


def feature_columns(history, with_transitions=False):
    cols = [f"lag{j}_{a}" for j in range(1, history + 1) for a in ATTRIBUTES]
    if with_transitions:
        if history < 2:
            raise ValueError("Transitions require at least two previous events")
        for j in range(1, history):
            cols += [f"change{j}_x", f"change{j}_y", f"change{j}_team"]
    return cols


def normalized_features(frame, history, with_transitions):
    """Return model inputs and their names, using fixed, non-learned scales."""
    base_cols = feature_columns(history)
    values = frame[base_cols].to_numpy(dtype=np.float32, copy=True)
    divisors = np.array([SCALES[a] for _ in range(history) for a in ATTRIBUTES],
                        dtype=np.float32)
    values /= divisors
    columns = list(base_cols)
    if with_transitions:
        additions = []
        for j in range(1, history):
            # Differences are between adjacent known PAST events only.
            dx = (frame[f"lag{j}_x"] - frame[f"lag{j+1}_x"]).to_numpy(dtype=np.float32) / 100
            dy = (frame[f"lag{j}_y"] - frame[f"lag{j+1}_y"]).to_numpy(dtype=np.float32) / 100
            team_switch = (frame[f"lag{j}_h"] != frame[f"lag{j+1}_h"]).to_numpy(dtype=np.float32)
            additions.extend([dx, dy, team_switch])
        values = np.column_stack([values] + additions).astype(np.float32)
        columns = feature_columns(history, True)
    if not np.isfinite(values).all():
        raise ValueError("Model features contain non-finite values")
    assert values.shape[1] == len(columns)
    return values.copy(), columns


def load_authors_mlp(script_path):
    spec = importlib.util.spec_from_file_location("authors_lem_training", script_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load original MLP: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLP


def train_and_score(train_frame, eval_frame, authors_script, epochs, batch_size, hidden, lr):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    MLP = load_authors_mlp(authors_script)
    train_labels = train_frame["e"].to_numpy(dtype=np.int64)
    eval_labels = eval_frame["e"].to_numpy(dtype=np.int64)
    classes = np.unique(train_labels)
    if not set(eval_labels).issubset(set(classes)):
        raise ValueError("Evaluation includes an event category absent from training")
    # Preserve the original integer class ID mapping for evaluation.
    class_index = {int(label): i for i, label in enumerate(classes)}
    train_targets = np.array([class_index[int(label)] for label in train_labels],
                             dtype=np.int64)
    y_train = torch.from_numpy(train_targets.copy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    print(f"Using {device}; train rows={len(train_frame):,}; evaluation rows={len(eval_frame):,}")
    for name, history, add_transitions in VARIANTS:
        # Reset for comparable, repeatable architectures and initializations.
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        x_train, columns = normalized_features(train_frame, history, add_transitions)
        x_eval, _ = normalized_features(eval_frame, history, add_transitions)
        model = MLP(input_size=x_train.shape[1], hidden_sizes=[hidden]*3,
                    output_size=len(classes), dropout_rate=0.3).to(device)
        loader = DataLoader(TensorDataset(torch.from_numpy(x_train), y_train),
                            batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            total, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(xb)
                seen += len(xb)
            if epoch in {0, epochs-1} or (epoch + 1) % 5 == 0:
                print(f"  {name}: epoch {epoch+1}/{epochs}, train loss={total/seen:.4f}", flush=True)
        model.eval()
        predicted_indices = []
        with torch.no_grad():
            for start in range(0, len(x_eval), batch_size):
                xb = torch.from_numpy(x_eval[start:start+batch_size]).to(device)
                predicted_indices.append(model(xb).argmax(dim=1).cpu().numpy())
        predictions = classes[np.concatenate(predicted_indices)]
        result = {"variant": name, "history_events": history,
                  "transition_features": add_transitions, "n_features": len(columns),
                  "test_or_validation_accuracy": float(accuracy_score(eval_labels, predictions)),
                  "test_or_validation_macro_f1": float(f1_score(eval_labels, predictions,
                                                       average="macro", zero_division=0))}
        rows.append(result)
        print(f"  {name}: accuracy={result['test_or_validation_accuracy']:.4f}, "
              f"macro F1={result['test_or_validation_macro_f1']:.4f}")
    return rows, str(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="real_wyscout_70_20_10")
    ap.add_argument("--authors-script", default="0001 Train Tabular LEMs.py")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=196)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--final-eval", action="store_true")
    args = ap.parse_args()
    if args.split == "val" and not args.final_eval:
        ap.error("Validation is sealed. Only use --split val --final-eval after freezing the model.")
    if min(args.epochs, args.batch_size, args.hidden) < 1 or args.learning_rate <= 0:
        ap.error("epochs, batch-size, hidden, and learning-rate must be positive")
    folder = Path(args.data)
    train_path = folder / "train_raw_events.csv"
    eval_path = folder / f"{args.split}_raw_events.csv"
    # No validation files are read in ordinary test mode.
    train_frame = build_event_history(pd.read_csv(train_path), max_history=5)
    eval_frame = build_event_history(pd.read_csv(eval_path), max_history=5)
    if train_frame.empty or eval_frame.empty:
        raise SystemExit("No five-event histories available. Use more matches/events.")
    results, device = train_and_score(train_frame, eval_frame,
                                       Path(args.authors_script), args.epochs,
                                       args.batch_size, args.hidden, args.learning_rate)
    report = {"data": str(folder), "source": "Public Wyscout 2017/18; authors MLP class",
              "exact_paper_reproduction": False, "dataset_split": args.split,
              "train_rows_common": len(train_frame), "evaluation_rows_common": len(eval_frame),
              "train_matches": int(train_frame.match_id.nunique()),
              "evaluation_matches": int(eval_frame.match_id.nunique()),
              "epochs": args.epochs, "hidden_sizes": [args.hidden]*3,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "random_seed": 42, "device": device, "results": results}
    output = folder / f"novel_context_{args.split}_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nCOMPARISON TABLE (identical eligible rows for all four variants)")
    print("Variant                    Features   Accuracy   Macro F1")
    for item in results:
        print(f"{item['variant']:<27}{item['n_features']:>8}   "
              f"{item['test_or_validation_accuracy']:>8.4f}   "
              f"{item['test_or_validation_macro_f1']:>8.4f}")
    print(f"Saved observed metrics to {output.resolve()}")
    print("This adapted experiment is not an exact reproduction of the parent paper.")


if __name__ == "__main__":
    main()
