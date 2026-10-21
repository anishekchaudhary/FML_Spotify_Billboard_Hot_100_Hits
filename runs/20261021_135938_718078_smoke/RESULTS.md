# Spotify results: smoke

This run predicts Spotify popularity, not measured Billboard chart membership.

Raw rows: 114,000; cleaned: 71,945; modeled: 8,000.
Training rows: 6,400; holdout rows: 1,600 (51 actual hits). Hit means popularity > 70.

All stages share a saved stratified holdout. Every estimator stores its own genre encoder, scaler, PCA and model. Preprocessing is fitted within training folds; genre training encodings are cross-fitted.
Winners are selected using cross-validation, never holdout scores. Training metrics use the final refitted encoder and are descriptive only.

## Measured results

### classification: RandomForestClassifier

Accuracy 89.38%; majority baseline 96.81%.
Hit precision 20.40%; recall 80.39%; F1 0.3254; average precision 0.2483.
Confusion matrix [[TN, FP], [FN, TP]]: [[1389, 160], [10, 41]].
Selection metric: hit_f1; CV score 0.2635.

### regression: Ridge

MAE 11.188; RMSE 15.510; R² 0.361; within ±5 points 36.00%.
Mean-prediction baseline: MAE 16.179; RMSE 19.415; R² -0.001.
Selection metric: neg_rmse; CV score -15.6774.

### regression_high: ElasticNet

MAE 3.018; RMSE 3.791; R² 0.141; within ±5 points 84.31%.
Mean-prediction baseline: MAE 3.286; RMSE 4.093; R² -0.001.
Selection metric: neg_rmse; CV score -4.5789.

## Combined workflow

Shortlisted 201 tracks from 1600 holdout tracks.
The high-popularity regressor scores all shortlisted tracks, including false positives. These metrics describe the shortlist, not every track or future Billboard success.
Shortlist regression: {"rows": 201, "mae": 28.148268362468336, "rmse": 38.4921767675666, "r2": -1.0831664971697457, "within_5_percent": 17.91044776119403}.

## Interpretation and limits

A majority-class baseline can have high accuracy while finding zero hits. Use hit precision, recall, F1 and average precision to judge discovery. Regression R² must be compared to a mean baseline. Within ±5 points is a tolerance metric, not classification accuracy.

The original saved RF classifier had 96.70% accuracy on a different random split. The original general-regression Lasso had RMSE 15.542 and R² 0.371; the high-popularity Lasso had RMSE 4.179 and R² 0.223. These are historical notebook outputs, not rerun results. Different samples, stratification, cross-fitted encoding, valid validation and selection criteria mean this run is not a direct reproduction of those numbers.

Random track splitting does not test future releases or independence between artists. Song characteristics alone omit marketing, audience and release timing. The high-popularity model is trained on known hits and may overestimate false positives.

This is a small-sample pipeline verification. Its model rankings and metrics are unstable; use the full-data run for the next experiment.

Open `metrics/*_summary.csv` for all models, `predictions/ranked_candidates.csv` for the shortlist, and `executed_notebooks/` for cell-by-cell outputs. Timing and hashes are recorded in `run_manifest.json`.
