"""Reproduce the preliminary, chronologically held-out Ridge comparison.

Run from repository root:
    python -m pip install pandas numpy scikit-learn
    python src/transfer_baseline.py

This is an exploratory transfer-prediction test, NOT a causal estimate or
a fully tuned model-selection study. It uses only pre-transfer predictors.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pl_outgoing_attacker_transfers.csv"
OUT = ROOT / "results" / "exploratory_transfer_ridge.json"
FEATURES = [
    "xg_p90_pre", "xa_p90_pre", "shots_p90_pre",
    "key_passes_p90_pre", "time_pre",
    "team_attack_strength_index_pre",
    "league_avg_team_xg_per_match_pre",
]
LEAGUES = ["Bundesliga", "Serie_A", "Ligue_1"]  # La_Liga reference
TARGET = "delta_expected_output_p90"


def score(truth, pred):
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "r2": float(r2_score(truth, pred)),
    }


def main():
    df = pd.read_csv(DATA)
    needed = FEATURES + ["league_post", "season_label_pre", "id", TARGET]
    missing = [name for name in needed if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    years = df["season_label_pre"].astype(str).str.slice(0, 4).astype(int)
    is_test = years >= 2022
    test = df.loc[is_test].copy()
    train = df.loc[~is_test].copy()
    # Remove any individual represented in later-season evaluation.
    train = train[~train["id"].isin(test["id"])].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split needs both train and test rows")
    if train["id"].isin(test["id"]).any():
        raise AssertionError("A player appears in both splits")

    def design(frame):
        x = frame[FEATURES].apply(pd.to_numeric, errors="raise").copy()
        for league in LEAGUES:
            x[f"destination_{league}"] = (
                frame["league_post"] == league
            ).astype(float)
        return x

    x_train, x_test = design(train), design(test)
    y_train = pd.to_numeric(train[TARGET], errors="raise")
    y_test = pd.to_numeric(test[TARGET], errors="raise")
    if not np.isfinite(x_train.to_numpy()).all() or not np.isfinite(x_test.to_numpy()).all():
        raise ValueError("Non-finite input values")
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=10.0).fit(scaler.transform(x_train), y_train)
    ridge_predictions = model.predict(scaler.transform(x_test))
    reference_predictions = np.full(len(test), float(y_train.mean()))
    report = {
        "data": str(DATA.relative_to(ROOT)),
        "evaluation": "exploratory chronological holdout, untuned",
        "target": TARGET,
        "pretransfer_features": FEATURES,
        "destination_indicators": LEAGUES,
        "split": "training years < 2022; test years >= 2022",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "ridge_alpha": 10.0,
        "mean_baseline": score(y_test, reference_predictions),
        "ridge": score(y_test, ridge_predictions),
        "limitations": [
            "Small sample and single split; no nested tuning or uncertainty intervals.",
            "Understat-based season pairs are not independently date verified.",
            "League scoring environment is not intrinsic league strength.",
            "Prediction does not identify a causal effect of transfer.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
