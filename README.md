> **For the professor's new-computer demonstration, follow [CLEAN_MACHINE_DEMO.md](CLEAN_MACHINE_DEMO.md). Run `python run_final.py` from the downloaded repository root. No third-party packages are needed.**\n\n# Context-Aware Transfer Analysis of Premier League Attackers

## Research question
Which Premier League attackers change their performance after moving to a different European domestic league, and which playing-environment variables are associated with that difference?

This public repository is the DS340W research project of Marc Otayek Chreim (Penn State, Fall 2026). The main *transfer* analysis and the separate *Wyscout next-event* experiment are distinct and should not be conflated.

## Main transfer dataset
- **[Transfer modeling dataset](data/pl_outgoing_attacker_transfers.csv):** 74 consecutive-season EPL-to-other-league attacker observations involving 71 unique players.
- **[Dataset documentation](DATASET_README.md)** and **[field dictionary](DATA_DICTIONARY.csv)** explain inputs, provenance, and transfer-pair construction.
- Destination counts: La Liga 27; Serie A 21; Bundesliga 13; Ligue 1 13.
- The 2025/26 player-stat supplement is available at [data/player_stats_2025_26.csv](data/player_stats_2025_26.csv).

The source is Understat-derived season-level statistical data. Consecutive-season moves are **candidate transfer pairs**, not independently verified transfer dates. Same-season ambiguous moves are separate. The league-average team xG variable is a *scoring-environment proxy*, not a direct measure of league strength.

## Reproduce the initial transfer prediction baseline
For the **required new-computer demonstration**, run only:

```bash
python run_final.py
```

No third-party packages are required for this authoritative transfer-results entry point. The optional scikit-learn implementation remains in `src/transfer_baseline.py` for comparison and requires `pip install -r requirements.txt`.

This script uses a **chronological holdout**, training on pre-transfer seasons before 2022–23 and testing on 2022–23 onward. It uses **pre-transfer predictors** and the known destination league only; post-transfer performance is the prediction target, not a feature. The model is Ridge regression with a fixed penalty (alpha = 10), compared against a training-mean baseline. The script writes `results/exploratory_transfer_ridge.json` when executed.

Preliminary figures, rounded (51 train / 23 test):

| Test metric | Training-mean baseline | Ridge |
|---|---:|---:|
| MAE | 0.1653 | 0.1574 |
| RMSE | 0.2075 | 0.2004 |
| R² | -0.0237 | 0.0453 |

**Interpretation:** modest preliminary predictive gain, not a validated transfer-causality claim. Results have not been assessed with nested temporal cross-validation or uncertainty intervals. Random Forest and XGBoost are methodological candidates, **not completed transfer-model benchmarks** here.

## Completed novel experiment: controlled context-feature ablation

The implemented novelty is a **same-split Ridge ablation**, evaluated on the
same 51 earlier training transfers and 23 later held-out transfers. The target
is change in expected attacking output per 90. Each variant uses the same
regularization (alpha=10) and scales its features using **training rows only**.

| Model | Feature count | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|---:|
| Train-mean reference | 0 | 0.1653 | 0.2075 | -0.0237 |
| **Pre-transfer player only** | 5 | **0.1545** | **0.1985** | **0.0631** |
| Player + team proxy | 6 | 0.1618 | 0.2040 | 0.0109 |
| Player + team + league context | 10 | 0.1574 | 0.2004 | 0.0453 |

**Actual finding:** contrary to the initial context-improvement hypothesis,
the player-only variant performed best on this small chronological holdout.
This **does not** demonstrate that real team/league context is irrelevant:
the available proxies are limited, and this is one small test split with no
significance test. No causal conclusions are supported.

Run `python run_final.py` for **both** the original verified reference
comparison and the new ablation. Results are written to
`results/reproduction_report.json` and `results/context_ablation.json`.
The entry point requires no pip installation.

Implementation: [ablation_experiment.py](ablation_experiment.py).
The [unit tests](test_run_final.py) verify the metric values and train/test
counts. [GitHub Actions](.github/workflows/reproducibility.yml) checks the
whole command on fresh Python versions.

## Separate supporting event-sequence experiment
The [large-events-model](large-events-model) folder adapts a published event-model architecture to public Wyscout 2017/18 England match events. Its novel input-history experiment compares 1, 3, and 5 previous events, optionally with transitions, on common train/test rows. Reported same-row test accuracies ranged from **0.6041** (one previous event) to **0.6322** (five events + transitions); macro F1 ranged from 0.3030 to 0.3639. The **10-match held-out validation set has not been evaluated**. Logs were transcribed from a separate local experiment rather than uploaded full-precision run artifacts. **Event prediction results do not establish individual improvement after a transfer.** See the folder's README for commands and scope.

## Research connections and future work
Three parent-paper directions inform the project:
1. Explainable expected-goals analysis (measurement and interpretation).
2. Team formation/player quality (context).
3. Large Events Models (temporal match-event context).

Future experiments: hierarchical mixed-effects regression; matched-player difference-in-differences; player-style clustering and destination-league interaction analysis.

## Reproducibility and limitations
- Recreate transfer source data with `python fetch_2025_26_stats.py` and `python build_dataset.py` as described in [DATASET_README.md](DATASET_README.md). Source availability and historical aggregation conventions may affect reproduction.
- Transfer timing, loans, season aggregates, position labels, and minutes selection require further auditing.
- No causal conclusions are warranted from raw pre/post comparisons.
- Repo contains partial notebook scaffolds and historical research stages; see scripts and metrics JSONs for implemented work.
