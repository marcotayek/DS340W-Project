"""Reproduce DS340W transfer results on a clean machine using Python only.

Run from the downloaded GitHub repository folder:
    python run_final.py

No packages or network calls are needed after downloading the repository ZIP.
The output is an exploratory prediction benchmark, not a causal transfer effect.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pl_outgoing_attacker_transfers.csv"
OUTPUT = ROOT / "results" / "reproduction_report.json"
FEATURES = (
    "xg_p90_pre", "xa_p90_pre", "shots_p90_pre", "key_passes_p90_pre",
    "time_pre", "team_attack_strength_index_pre",
    "league_avg_team_xg_per_match_pre",
)
LEAGUES = ("Bundesliga", "Serie_A", "Ligue_1")  # La_Liga is reference
EXPECTED = {
    "rows": 74, "players": 71, "train": 51, "test": 23,
    "positive": 52, "mean_change": .12927008694265366,
    "baseline_mae": .1653201198345859,
    "baseline_rmse": .20750637451378526,
    "baseline_r2": -.023687579064719966,
    "ridge_mae": .15742955950256807,
    "ridge_rmse": .20039723448548089,
    "ridge_r2": .045253672760217745,
}


def solve(a, b):
    """Gaussian elimination with partial pivoting for the Ridge normal equations."""
    n = len(b)
    m = [[*map(float, a[i]), float(b[i])] for i in range(n)]
    for j in range(n):
        p = max(range(j, n), key=lambda k: abs(m[k][j]))
        if abs(m[p][j]) < 1e-12:
            raise ValueError("Singular coefficient matrix")
        m[j], m[p] = m[p], m[j]
        f = m[j][j]
        for k in range(j, n+1):
            m[j][k] /= f
        for i in range(n):
            if i == j:
                continue
            f = m[i][j]
            for k in range(j, n+1):
                m[i][k] -= f*m[j][k]
    return [m[i][-1] for i in range(n)]


def scores(y, pred):
    err = [a-b for a, b in zip(y, pred)]
    total = sum(x*x for x in err)
    mu = sum(y)/len(y)
    denom = sum((a-mu)**2 for a in y)
    return {
        "mae": sum(map(abs, err))/len(y),
        "rmse": math.sqrt(total/len(y)),
        "r2": 1-total/denom if denom else None,
    }


def run(data=DATA, output=OUTPUT, check=True):
    if not data.is_file():
        raise FileNotFoundError(f"Missing {data}. Download the full repository ZIP.")
    with data.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Empty dataset")
    required = {"id", "season_label_pre", "league_post",
                "delta_expected_output_p90", *FEATURES}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for row in rows:
        row["year"] = int(row["season_label_pre"][:4])
        row["target"] = float(row["delta_expected_output_p90"])
        row["numeric"] = [float(row[k]) for k in FEATURES]
        if not all(math.isfinite(v) for v in [row["target"], *row["numeric"]]):
            raise ValueError("Non-finite data")
        if row["league_post"] not in {*LEAGUES, "La_Liga"}:
            raise ValueError(f"Unexpected destination league: {row['league_post']}")

    test = [r for r in rows if r["year"] >= 2022]
    test_ids = {r["id"] for r in test}
    train = [r for r in rows if r["year"] < 2022 and r["id"] not in test_ids]
    if not train or not test or ({r["id"] for r in train} & test_ids):
        raise ValueError("Invalid or overlapping chronological partitions")

    def raw(r):
        return r["numeric"] + [float(r["league_post"] == league)
                                for league in LEAGUES]

    train_x = [raw(r) for r in train]
    test_x = [raw(r) for r in test]
    cols = len(train_x[0])
    means = [sum(r[j] for r in train_x)/len(train_x) for j in range(cols)]
    stds = [(sum((r[j]-means[j])**2 for r in train_x)/len(train_x))**.5
            for j in range(cols)]
    stds = [v or 1.0 for v in stds]

    def standardize(r):
        return [1.] + [(r[j]-means[j])/stds[j] for j in range(cols)]

    train_x = [standardize(r) for r in train_x]
    test_x = [standardize(r) for r in test_x]
    y_train = [r["target"] for r in train]
    y_test = [r["target"] for r in test]
    k = cols+1
    gram = [[sum(r[i]*r[j] for r in train_x)
             + (10. if i == j and i else 0.)
             for j in range(k)] for i in range(k)]
    rhs = [sum(r[i]*y for r, y in zip(train_x, y_train))
           for i in range(k)]
    beta = solve(gram, rhs)
    ridge_pred = [sum(x*v for x, v in zip(r, beta)) for r in test_x]
    mean_train = sum(y_train)/len(y_train)
    base = scores(y_test, [mean_train]*len(test))
    ridge = scores(y_test, ridge_pred)

    report = {
        "rows": len(rows), "players": len({r["id"] for r in rows}),
        "train": len(train), "test": len(test),
        "positive": sum(r["target"] > 0 for r in rows),
        "mean_change": sum(r["target"] for r in rows)/len(rows),
        "baseline": base, "ridge": ridge,
        "destination_counts": {name: sum(r["league_post"] == name for r in rows)
                               for name in ("La_Liga", "Serie_A", "Bundesliga", "Ligue_1")},
        "protocol": "Train before 2022-23; evaluate 2022-23 to 2024-25; Ridge alpha=10",
        "limitations": "Season-pair candidates, no verified transfer dates; not causal",
    }
    flat = {key: report[key] for key in ("rows", "players", "train", "test", "positive", "mean_change")}
    flat.update({f"{name}_{key}": value for name, result in
                 (("baseline", base), ("ridge", ridge))
                 for key, value in result.items()})
    print("DS340W TRANSFER-PREDICTION REPRODUCTION")
    print(f"Transfers: {report['rows']} | Players: {report['players']} | "
          f"Positive changes: {report['positive']} | Mean change: {report['mean_change']:.4f}")
    print(f"Train/test: {len(train)}/{len(test)}")
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for name, values in (("Mean baseline", base), ("Ridge alpha=10", ridge)):
        print(f"{name:<20}{values['mae']:>10.4f}{values['rmse']:>10.4f}{values['r2']:>10.4f}")
    if check:
        for key, expected in EXPECTED.items():
            if not math.isclose(flat[key], expected, abs_tol=0.00005, rel_tol=0):
                raise AssertionError(f"Paper mismatch for {key}: {flat[key]} != {expected}")
        print(f"PASS: {len(EXPECTED)} checks match published paper numbers.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Saved:", output)
    return report


if __name__ == "__main__":
    run()
    from ablation_experiment import run_ablation
    run_ablation()
