# Large Events Model — Public Wyscout Adaptation

## Purpose
This folder documents an **adapted reproduction experiment** related to *Towards a Foundation Large Events Model for Soccer* (Mendes-Neves et al., 2024; DOI: https://doi.org/10.1007/s10994-024-06606-y).

We use the **original authors' MLP class** from `0001 Train Tabular LEMs.py`, but use a different public Wyscout dataset, simplified previous-event features, target mappings, and training procedure. **These results are not an exact reproduction of the published benchmarks.**

## Data
Public Wyscout 2017–18 event dataset by Pappalardo et al. (collection: https://doi.org/10.6084/m9.figshare.c.4415000). Download the **Events** and **Matches** datasets and extract `events_England.json` and `matches_England.json`. The conversion below uses 80 English league matches, split by match (seed 42): 64 training matches and 16 validation matches.

- Training: 104,953 usable next-event examples
- Validation: 27,033 usable next-event examples

## Reproduce on Windows (Python 3.12)
From inside this folder, install dependencies:

```powershell
py -3.12 -m pip install -r requirements.txt
```

Convert the downloaded files (adjust paths to your computer):

```powershell
py -3.12 convert_public_wyscout.py --events "C:\path\to\events_England.json" --matches "C:\path\to\matches_England.json" --output real_wyscout --max-matches 80
```

Train and evaluate the gradient-boosting baseline:

```powershell
py -3.12 train_real_wyscout.py --data real_wyscout
```

Train and evaluate the authors' MLP architecture adapted to these targets:

```powershell
py -3.12 train_authors_mlp_real.py --data real_wyscout --epochs 30
```

The scripts write metrics into `real_wyscout/`. A CUDA-capable GPU can accelerate the MLP; CPU is supported.

## Validation results

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

The MLP's event accuracy advantage is only 0.05 percentage points; the baseline performs better on the other reported metrics.

## Files
- `convert_public_wyscout.py`: maps public Wyscout JSON into match-separated CSVs for next-event prediction.
- `train_real_wyscout.py`: histogram gradient boosting baseline.
- `train_authors_mlp_real.py`: adapted training using the original authors' MLP class.
- `0001 Train Tabular LEMs.py`: authors' original training script containing the MLP class.
- `baseline_metrics.json` and `authors_mlp_30epochs_metrics.json`: saved metrics from the experiment.

## Limitations and attribution
Public Wyscout event names and tags do not perfectly match the paper's Wyscout V3 schema. Event categories and success-related fields are approximated, and the simplified model uses only the previous event's six attributes. We do not claim equivalence with the paper's full training setup, dataset, or reported performance. The original model implementation belongs to its authors; see https://github.com/nvsclub/LargeEventsModel and retain their original license/attribution when redistributing their source.

**Dataset citation:** Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data, https://doi.org/10.1038/s41597-019-0247-7.
