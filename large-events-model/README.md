# Large Events Model — Public Wyscout Adaptation

## Purpose
This folder documents an **adapted reproduction experiment** related to *Towards a Foundation Large Events Model for Soccer* (Mendes-Neves et al., 2024; DOI: https://doi.org/10.1007/s10994-024-06606-y).

We use the **original authors' MLP class** from `0001 Train Tabular LEMs.py`, but use a different public Wyscout dataset, simplified previous-event features, target mappings, and training procedure. **These results are not an exact reproduction of the published benchmarks.**

## Data
Public Wyscout 2017–18 event dataset by Pappalardo et al. (collection: https://doi.org/10.6084/m9.figshare.c.4415000). Download the **Events** and **Matches** datasets and extract `events_England.json` and `matches_England.json`. The **new protocol** uses 100 English league matches, split by match into **70 train / 20 test / 10 validation** (seed 42 for the development-pool shuffle). Matches 91–100 in sorted match-ID order were **not part of the earlier 80-match experiment** and are reserved as a fresh holdout. Exact event counts will be printed when you run preprocessing.

- **Training (70 matches):** fit the models.
- **Test (20 matches):** debug, compare models, and choose hyperparameters.
- **Validation (10 matches):** sealed until model selection and code are frozen. Only run final evaluation once.

The old 80/20 experiment and its JSON files remain historical results, **not** results from this new three-way split.

## Reproduce on Windows (Python 3.12)
From inside this folder, install dependencies:

```powershell
py -3.12 -m pip install -r requirements.txt
```

Convert the downloaded files (adjust paths to your computer):

```powershell
py -3.12 convert_public_wyscout.py --events "C:\path\to\events_England.json" --matches "C:\path\to\matches_England.json" --output real_wyscout_70_20_10 --max-matches 100
```

Train and evaluate the gradient-boosting baseline:

```powershell
py -3.12 train_real_wyscout.py --data real_wyscout_70_20_10 --split test
```

Train and evaluate the authors' MLP architecture adapted to these targets:

```powershell
py -3.12 train_authors_mlp_real.py --data real_wyscout_70_20_10 --epochs 30 --split test
```

The scripts write test metrics into `real_wyscout_70_20_10/`. A CUDA-capable GPU can accelerate the MLP; CPU is supported. `split_manifest.json` records the exact match assignments.

**Only after debugging, tuning, and choosing the final model**, run the following *once* and report both test and final validation scores:

```powershell
py -3.12 train_real_wyscout.py --data real_wyscout_70_20_10 --split val --final-eval
py -3.12 train_authors_mlp_real.py --data real_wyscout_70_20_10 --epochs 30 --split val --final-eval
```

Do not use validation scores to tune the model. Note: the MLP command retrains with the chosen configuration before final evaluation; for a strictly frozen-model evaluation, retain the test-stage model checkpoints and load those weights instead. The `--final-eval` flag is a deliberate procedural guard, not a technical access restriction.

## New 70/20/10 experiment — observed test results (November 2026)

The updated scripts were run locally on 100 matches. The split was **70 training / 20 test / 10 validation matches**. Preprocessing yielded **115,415 training**, **33,223 test**, and **17,264 validation** usable next-event examples. The validation set was generated but **has not been evaluated**.

| Test metric | Gradient boosting baseline | Adapted authors' MLP (30 epochs) |
|---|---:|---:|
| Event accuracy (higher better) | 0.6024 | **0.6036** |
| Event macro F1 (higher better) | **0.3205** | 0.3129 |
| X normalized MAE (lower better) | **0.1727** | 0.1753 |
| X R² (higher better) | **0.2287** | 0.2037 |
| Y normalized MAE (lower better) | **0.2351** | 0.2399 |
| Y R² (higher better) | **0.1948** | 0.1671 |
| Time MAE (lower better) | **0.5518** | 0.5685 |
| Time R² (higher better) | **0.4321** | 0.4013 |

The adapted MLP slightly improved event accuracy by 0.12 percentage points, but the baseline did better on the other listed metrics. The MLP completed all 30 epochs for each of its four prediction targets on CUDA.

**Provenance:** These numbers were transcribed from the local PowerShell logs and stored in [`test_results_70_20_10.json`](test_results_70_20_10.json). The original generated `baseline_test_metrics.json`, `authors_mlp_test_metrics.json`, and `split_manifest.json` remain on the local machine and have **not** been uploaded. The match-level assignments cannot be audited from the summary alone. The test results are not final validation results.

## Novel contribution — Five-event temporal context experiment

The modular experiment [`novel_context_experiment.py`](novel_context_experiment.py) modifies the **input feature block** of the authors' MLP architecture. Rather than using only one preceding event, it compares these four variants on the **same training and test event rows**:

| Variant | Previous events | Input features | New information |
|---|---:|---:|---|
| Previous 1 | 1 | 6 | Original one-event attributes; rerun on common rows |
| Previous 3 | 3 | 18 | Three chronological event attribute blocks |
| Previous 5 | 5 | 30 | Five chronological event attribute blocks |
| Previous 5 + transitions | 5 | 42 | x/y changes and team switches between past events |

Each event has the six available attributes: team-side indicator, event category, x, y, elapsed-interval bucket, and coarse success indicator. No unavailable player, league, injury, formation, or transfer variables are fabricated. All history is constructed within a match. For fair comparison, all four variants drop the first five events of every match and therefore predict **identical targets**. The training recipe is the same authors' MLP (three 196-unit hidden layers, dropout 0.3, Adam, seed 42, 30 epochs). Comparison metrics are next-event accuracy and macro F1.

**Check the new feature logic first (no real dataset or GPU needed):**

```powershell
py -3.12 -m unittest -v test_novel_context.py
```

These tests verify that lagged events cannot cross match boundaries, all four variants use identical eligible rows, and transition features contain only previous-event information. They are feature-construction tests, not a claim that training finished.

**Run from inside this folder (PowerShell):**

```powershell
py -3.12 novel_context_experiment.py --data real_wyscout_70_20_10 --epochs 30 --split test
```

The program prints a four-row table and saves `real_wyscout_70_20_10/novel_context_test_metrics.json`. **The experiment was run locally and the observed results are reported in the following section.** The earlier one-event MLP result (accuracy 0.6036, macro F1 0.3129) uses a slightly larger sample and is a historical reference, **not** a fair same-row comparison. The 10-match validation split remains sealed; test mode does not even open its CSV. Final validation is guarded by `--split val --final-eval`.

If additional event history improves macro F1, it supports the limited interpretation that recent game events add predictive information in this public-data adaptation. It does **not** directly establish whether an attacker improved after transferring. The eventual transfer study needs separate player-season, league, and role context.

## Novel contribution — observed four-way test comparison (November 15, 2026)

The new context-history experiment **ran successfully on CUDA** for 30 epochs per variant. The same **115,135 training examples** and **33,143 test examples** (after excluding the first five events per match) were used for all four models. The 10-match validation set remains **unevaluated**.

| Model variant | Input features | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Previous 1 event | 6 | 0.6041 | 0.3030 |
| Previous 3 events | 18 | 0.6306 | 0.3636 |
| Previous 5 events | 30 | 0.6299 | 0.3562 |
| Previous 5 events + transitions | 42 | **0.6322** | **0.3639** |

Relative to the **one-event model trained and tested on the same rows**, the five-event + transitions model improved event accuracy by **2.81 percentage points** (60.41% to 63.22%) and macro F1 by **0.0609** (0.3030 to 0.3639). The three-event model was close to the best (63.06% accuracy, 0.3636 macro F1) with fewer features. More history was **not** uniformly better: five events without transitions scored below three events on both measures. These observations motivate a model-complexity tradeoff rather than an unconditional claim that more features always help.

**Source:** User-provided PowerShell output from `py -3.12 novel_context_experiment.py --data real_wyscout_70_20_10 --epochs 30 --split test`. A faithful, rounded transcript is stored in [`novel_context_test_metrics.json`](novel_context_test_metrics.json); the original full-precision locally generated JSON file remains on the Windows PC and has not been uploaded. These are **adapted public-data test results, not a replication of the parent paper's benchmark and not held-out final validation results**. Because these are single-seed runs, they do not yet establish robustness across random initializations or matches.

## Historical results — OLD 80/20 experiment (not new 70/20/10 scores)

| Metric | Gradient boosting | Adapted authors' MLP (30 epochs) |
|---|---:|---:|
| Event accuracy (higher better) | 0.6019 | **0.6024** |
| Event macro F1 (higher better) | **0.3268** | 0.2859 |
| X MAE, normalized 0–1 (lower better) | **0.1727** | 0.1768 |
| X R² (higher better) | **0.2290** | 0.2024 |
| Y MAE, normalized 0–1 (lower better) | **0.2380** | 0.2421 |
| Y R² (higher better) | **0.1958** | 0.1661 |
| Time MAE (script units, lower better) | **0.5484** | 0.5824 |
| Time R² (higher better) | **0.4180** | 0.3915 |

The MLP's event accuracy advantage was only 0.05 percentage points in the **old experiment**; the baseline performed better on the other reported metrics. **Do not report these as new three-way-split test or validation results.**

## Files
- `convert_public_wyscout.py`: maps public Wyscout JSON into match-separated CSVs for next-event prediction.
- `train_real_wyscout.py`: histogram gradient boosting baseline.
- `train_authors_mlp_real.py`: adapted training using the original authors' MLP class.
- `0001 Train Tabular LEMs.py`: authors' original training script containing the MLP class.
- `test_results_70_20_10.json`: transcribed original baseline and MLP test metrics (not original local output files).
- `novel_context_test_metrics.json`: observed four-variant novel-approach comparison transcribed from the PowerShell output.
- `baseline_metrics.json` and `authors_mlp_30epochs_metrics.json`: saved metrics from the historical 80/20 experiment.

## Limitations and attribution
Public Wyscout event names and tags do not perfectly match the paper's Wyscout V3 schema. Event categories and success-related fields are approximated, and the simplified model uses only the previous event's six attributes. We do not claim equivalence with the paper's full training setup, dataset, or reported performance. The original model implementation belongs to its authors; see https://github.com/nvsclub/LargeEventsModel and retain their original license/attribution when redistributing their source.

**Dataset citation:** Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data, https://doi.org/10.1038/s41597-019-0247-7.
