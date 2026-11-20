# Clean-machine DS340W demonstration (must be screen-recorded)

**This is the exact procedure for a brand-new PSU computer.** It reproduces
the **main Premier League outgoing-attacker transfer results in the paper**.
The primary command reproduces both the original baseline AND the context-feature ablation using only Python's standard library; there is no package
installation, API token, GPU, local Windows path, or external dataset download
beyond the GitHub ZIP itself.

## Before recording (on your own machine)
1. Confirm the public repository contains `run_final.py` and
   `data/pl_outgoing_attacker_transfers.csv`.
2. Confirm your final research paper quotes these *transfer* results:
   74 observations, 71 unique players, 52 positive expected-output changes;
   51 train and 23 test;
   **mean baseline** MAE 0.1653, RMSE 0.2075, R² -0.0237;
   **Ridge** MAE 0.1574, RMSE 0.2004, R² 0.0453.
3. Distinguish the separately documented Wyscout next-event experiment.
   Its recorded historical scores are NOT reproduced by this command and must
   not be misrepresented as independently replicated on the clean machine.

## During the recording on a computer NEVER used for development

1. Log in to your PSU account and start PSU Zoom. Turn your webcam on and
   share the full screen. **Start recording before downloading any files.**
2. Open https://github.com/marcotayek/DS340W-Project
3. Click **Code > Download ZIP**. Show the download in the Zoom recording.
4. Extract the downloaded ZIP, then open the extracted
   `DS340W-Project-main` directory. You should see `run_final.py` and `data`.
5. Open a terminal in that folder (Windows: click File Explorer address bar,
   type `cmd`, and press Enter).
6. Run the command appropriate to the lab computer:

   **Windows**
   ```bat
   py -3 run_final.py
   ```
   If `py` is not installed, use `python run_final.py`.

   **macOS / Linux**
   ```bash
   python3 run_final.py
   ```

   Python 3.9+ is recommended. Check with `py -3 --version`,
   `python --version`, or `python3 --version`. If there is no Python
   installed, ask your lab administrator for a machine with Python. Do not
   claim that it was tested on the new machine until the recording passes.

7. (Optional stronger proof) Run the unit tests in the same terminal:\n\n   **Windows**: `py -3 -m unittest -v test_run_final.py`\n\n   **macOS / Linux**: `python3 -m unittest -v test_run_final.py`\n\n   The tests check both the Ridge linear solver and reproducibility of the paper's transfer results.\n\n8. Show the terminal output and the new
   `results/reproduction_report.json` file. The script verifies every
   published transfer dataset count and metric within tolerance and displays:

   ```text
   Transfers: 74 | Players: 71 | Positive changes: 52 | Mean change: 0.1293
   Train/test: 51/23
   Model                      MAE      RMSE        R2
   Mean baseline           0.1653    0.2075   -0.0237
   Ridge alpha=10          0.1574    0.2004    0.0453
   PASS: 12 checks match published paper numbers.
   ```

9. Explain the key lines: load the dataset, select ONLY pre-transfer features,
   make the chronological split, fit normalization ONLY on the training data,
   solve Ridge regression with a fixed penalty, compare against a training-mean
   prediction, calculate MAE/RMSE/R², and verify the paper values.
10. Show the matching table in your paper. Stop the Zoom recording.
11. Upload the Zoom recording to Kaltura as **Unlisted** and submit its link.

## Troubleshooting
- **'py' is not recognized**: use `python run_final.py` or
  `python3 run_final.py`.
- **No Python on the lab machine**: switch to a lab machine with Python.
- **Dataset missing**: your current working folder is wrong, or you downloaded
  one file rather than the complete ZIP. Use Code > Download ZIP again.
- **Paper-mismatch assertion**: do not hide it. The CSV or code differs
  from the experiment used in the paper; restore the exact repository release.
- **Why no XGBoost installation?** It is not used to generate the published
  transfer benchmark. The historically adapted Wyscout model requires its
  own separate third-party data and environment; see its folder README.

## Research scope
This experiment measures predictive association for **Premier League attackers
moving to La Liga, Serie A, Bundesliga and Ligue 1**. The league scoring
environment proxy is not a formal league-strength rating; pre/post comparison
does not prove causal improvement.

## NEW: what the novelty demonstration must also show

The same \`python run_final.py\` command automatically runs the three-way
context ablation, after the original transfer-baseline checks. The paper's
observed held-out values (51 training / 23 test) are:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Training-mean reference | 0.1653 | 0.2075 | -0.0237 |
| Player-only Ridge (5 features) | 0.1545 | 0.1985 | 0.0631 |
| Player + team (6 features) | 0.1618 | 0.2040 | 0.0109 |
| Full context (10 features) | 0.1574 | 0.2004 | 0.0453 |

After running, open \`results/context_ablation.json\` and show it matches the
**updated** paper. Explain that adding context proxies did **not** improve
prediction over player-only Ridge on this small holdout. This is the study's
implemented novelty: a controlled feature ablation, not a claim of a causal
transfer effect. The unit-test command also checks all three ablation scores.
