# Pipeline validation and small-run findings

Completed on 21 October 2026, Asia/Kolkata. Full-data model training has **not** been started.

## What was executed

- Python 3.14.7 virtual environment created in `.venv`; required packages installed and exact versions saved in `requirements.lock.txt`.
- All 13 direct dependency imports pass. `pip check` reports no broken requirements.
- XGBoost's missing native dependency was found and installed: Homebrew `libomp` 23.1.4. Its resolved path and hash are in each final run's `environment.json`.
- The project Jupyter kernel is registered under `.venv/share/jupyter/kernels/ml-spotify`. Automated runs use a run-local specification pointing at this same interpreter.
- Seven boundary tests pass: shared disjoint holdout, duplicate split rejection, unknown partition rejection, empty shortlist, false positives retained in shortlist evaluation, single-row R² handling and invalid worker count.
- All five corrected notebooks completed on 8,000 stratified tracks, with two CV folds and two candidate settings per family: 10 classifiers, 8 general regressors and 8 high-popularity regressors, totaling 130 model fits including refits.
- Saved model reloads reproduce predictions. The artifact verifier passes for all notebooks, model families, holdout predictions and the combined shortlist.
- A second successful small run produced exactly the same prediction CSV files as the first. Its elapsed time was 38.2 seconds; this is not a full-run runtime estimate.
- Original datasets and root notebooks were not modified. All six backup hashes were verified; the only pre-existing file edited was `README.md`.

Final small-run directory:

`runs/20261021_135938_718078_smoke/`

Open [the generated results report](runs/20261021_135938_718078_smoke/RESULTS.md), [the pipeline log](runs/20261021_135938_718078_smoke/pipeline.log), or the five notebooks in that directory's `executed_notebooks/`.

The first successful run is retained in `runs/20261021_135822_534186_smoke/`. Earlier failed setup attempts are also retained as audit records. Those attempts identified the missing OpenMP runtime and the sandbox restriction on Jupyter's local sockets. The successful smoke runs used authorized execution outside that sandbox. The final run closes kernel connections cleanly. The smoke MLP reaches its deliberately small 100-iteration cap; its convergence warning is recorded in the logs, and its results should be interpreted accordingly.

## What the small-run results mean

Cleaning produced 71,945 tracks and reproduced the original processed CSV values. The small sample used 6,400 training tracks and 1,600 holdout tracks, including 51 actual hits. A hit means Spotify popularity > 70.

| Stage | Model selected using training CV | Holdout result |
|---|---|---|
| Classification | Random Forest | Hit precision 20.40%; recall 80.39%; F1 0.3254; accuracy 89.38% |
| General regression | Ridge | MAE 11.188; RMSE 15.510; R² 0.361; 36.00% within ±5 points |
| Known-hit regression | ElasticNet | MAE 3.018; RMSE 3.791; R² 0.141; 84.31% within ±5 points |
| Combined shortlist | Selected classifier → known-hit regressor | 201 shortlisted tracks; MAE 28.148; RMSE 38.492; R² −1.083 |

The classifier found 41 of 51 actual hits, missed 10, and incorrectly shortlisted 160 non-hits. Thus only about one song in five on its shortlist was actually a hit. Its overall accuracy is lower than the 96.81% majority-class baseline, but that baseline finds zero hits. Recall and precision expose this tradeoff more clearly than accuracy alone.

The general regressor improved on its training-mean baseline: mean-baseline MAE was 16.179 and RMSE was 19.415. Its R² of 0.361 indicates useful signal within this holdout, with substantial error remaining.

Predicting only known hits is easier in absolute points because popularity is restricted to 71–100. The known-hit model's MAE was 3.018 versus its mean baseline's 3.286. The modest R² of 0.141 shows why an 84.31% within-±5 score should not be described as strong general predictive accuracy.

The combined workflow estimates popularity using a regressor trained only on known hits. Its shortlist contains many lower-popularity false positives, so the scores overestimate many songs. Shortlist MAE rises to 28.148; the high-popularity mean baseline is similarly poor at 28.435. This is an observed limitation of the two-stage approach. Improving classifier precision or changing the scoring strategy would be a separate modeling experiment; the pipeline records this weakness rather than filtering it out of evaluation.

These results are a small-sample workflow verification. They do not establish the winning full-data model or predict actual Billboard membership. The original saved notebook results used different splits, encoding, tuning and selection, so comparisons are descriptive rather than controlled.

## Prepared full run

The final full-profile preflight passed:

`runs/20261021_140018_787381_full/run_manifest.json`

Its status is `checked_only` and its stage list is empty: no notebooks or model training were executed. The full configuration uses all cleaned tracks, five-fold cross-validation and up to 30 candidate settings per model family, totaling 3,746 model fits including refits. It uses two search workers and one native thread per worker. Full-data SVC and ensemble searches may take hours; small-run timing is not a reliable extrapolation.

Start it yourself from a terminal:

```bash
cd /Users/anishech/Projects/ML_Spotify
caffeinate -i .venv/bin/python -u run_pipeline.py --profile full
```

The printed run directory contains `pipeline.log` for progress and `RESULTS.md` when complete. All run folders are separate and are ignored by Git. See `configs/original_budget.json` only if you deliberately want the original, much larger search budgets with corrected validation; that profile was not run.

After completion:

```bash
.venv/bin/python verify_run.py runs/<your-full-run>
```

Share the run folder path, or `RESULTS.md`, `run_manifest.json`, the three `metrics/*_summary.csv` files and `metrics/combined.json`. If execution fails, share `logs/failure.log` and `pipeline.log` from that run.
