"""Reproducible contextual-feature ablation for EPL outgoing attacker transfers.

Only pre-transfer measurements and the destination league are used. All three
Ridge models use the same training years, held-out years, scaler training rows,
regularization alpha=10, and test-set metrics. This isolates incremental
predictive value of available context, NOT the causal effect of changing league.

Run from repository root: python run_final.py
Or standalone: python ablation_experiment.py
No third-party dependencies.
"""
import csv
import json
import math
from pathlib import Path
from run_final import DATA, ROOT, solve, scores

PLAYER = ("xg_p90_pre", "xa_p90_pre", "shots_p90_pre",
          "key_passes_p90_pre", "time_pre")
TEAM = ("team_attack_strength_index_pre",)
LEAGUE = ("league_avg_team_xg_per_match_pre",)
DESTINATIONS = ("Bundesliga", "Serie_A", "Ligue_1")
TARGET = "delta_expected_output_p90"
OUT = ROOT / "results" / "context_ablation.json"
ALPHA = 10.0


def load_data(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    needed = {*PLAYER, *TEAM, *LEAGUE, "id", "season_label_pre",
              "league_post", TARGET}
    if not rows or needed.difference(rows[0]):
        raise ValueError("Missing transfer observations or required CSV columns")
    for row in rows:
        row["_year"] = int(row["season_label_pre"][:4])
        for c in (*PLAYER, *TEAM, *LEAGUE, TARGET):
            row[c] = float(row[c])
            if not math.isfinite(row[c]):
                raise ValueError("Nonfinite value in " + c)
        if row["league_post"] not in (*DESTINATIONS, "La_Liga"):
            raise ValueError("Unknown league " + row["league_post"])
    test = [r for r in rows if r["_year"] >= 2022]
    test_ids = {r["id"] for r in test}
    train = [r for r in rows if r["_year"] < 2022
             and r["id"] not in test_ids]
    if not train or not test or {r["id"] for r in train} & test_ids:
        raise ValueError("Bad chronological split / player overlap")
    return rows, train, test


def design(row, group):
    columns = PLAYER + (TEAM if group in ("player_team", "full_context") else ())
    if group == "full_context":
        columns += LEAGUE
    raw = [row[x] for x in columns]
    if group == "full_context":
        raw += [float(row["league_post"] == dst) for dst in DESTINATIONS]
    return raw


def ridge_fit(train, test, group):
    raw_train = [design(r, group) for r in train]
    raw_test = [design(r, group) for r in test]
    size = len(raw_train[0])
    means = [sum(x[j] for x in raw_train) / len(raw_train)
             for j in range(size)]
    stds = [(sum((x[j]-means[j])**2 for x in raw_train)
             / len(raw_train))**.5 for j in range(size)]
    stds = [x or 1 for x in stds]
    def normalize(x):
        return [1.0] + [(x[j]-means[j])/stds[j] for j in range(size)]
    X = [normalize(x) for x in raw_train]
    Q = [normalize(x) for x in raw_test]
    y = [r[TARGET] for r in train]
    k = size+1
    gram = [[sum(row[i]*row[j] for row in X)
             + (ALPHA if i == j and i != 0 else 0)
             for j in range(k)] for i in range(k)]
    rhs = [sum(x[i]*t for x,t in zip(X, y)) for i in range(k)]
    coef = solve(gram, rhs)
    pred = [sum(a*b for a,b in zip(row, coef)) for row in Q]
    return scores([r[TARGET] for r in test], pred), size


def run_ablation(data=DATA, output=OUT, verbose=True):
    rows, train, test = load_data(data)
    actual = [r[TARGET] for r in test]
    average = sum(r[TARGET] for r in train) / len(train)
    mean_scores = scores(actual, [average]*len(test))
    variants = {}
    for name in ("player_only", "player_team", "full_context"):
        val, count = ridge_fit(train, test, name)
        variants[name] = {"features": count, **val}
    report = {
        "sample_size": len(rows), "train_rows": len(train), "test_rows": len(test),
        "split": "pre-2022 training, 2022 onward test; no repeated players",
        "target": TARGET, "algorithm": "Ridge L2 alpha=10.0, train-fitted scaling",
        "feature_sets": {
            "player_only": list(PLAYER),
            "player_team": list(PLAYER + TEAM),
            "full_context": list(PLAYER + TEAM + LEAGUE)
                + ["destination_" + d for d in DESTINATIONS],
        },
        "mean_baseline": mean_scores, "ridge_variants": variants,
        "scope": "Exploratory single-holdout association; not causal; small sample",
    }
    if verbose:
        print("\nCONTEXT ABLATION (same held-out transfer observations)")
        print(f"{'Variant':<21}{'Features':>10}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
        print(f"{'Train mean':<21}{0:>10d}{mean_scores['mae']:>10.4f}"
              f"{mean_scores['rmse']:>10.4f}{mean_scores['r2']:>10.4f}")
        for name, v in variants.items():
            print(f"{name:<21}{v['features']:>10d}{v['mae']:>10.4f}"
                  f"{v['rmse']:>10.4f}{v['r2']:>10.4f}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if verbose:
        print("Ablation results saved:", output)
    return report


if __name__ == "__main__":
    run_ablation()
