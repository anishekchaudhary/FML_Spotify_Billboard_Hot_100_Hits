# Spotify results: full

This run predicts Spotify popularity, not measured Billboard chart membership.

Raw rows: 114,000; cleaned: 71,945; modeled: 71,945.
Training rows: 57,556; holdout rows: 14,389 (456 actual hits). Hit means popularity > 70.

All stages share a saved stratified holdout. Every estimator stores its own genre encoder, scaler, PCA and model. Preprocessing is fitted within training folds; genre training encodings are cross-fitted.
Winners are selected using cross-validation, never holdout scores. Training metrics use the final refitted encoder and are descriptive only.

## Measured results

### classification: GaussianNB

Accuracy 94.63%; majority baseline 96.83%.
Hit precision 25.65%; recall 36.62%; F1 0.3017; average precision 0.2129.
Confusion matrix [[TN, FP], [FN, TP]]: [[13449, 484], [289, 167]].
Selection metric: hit_f1; CV score 0.3257.

### regression: XGBRegressor

MAE 11.082; RMSE 15.611; R² 0.367; within ±5 points 37.10%.
Mean-prediction baseline: MAE 16.274; RMSE 19.615; R² -0.000.
Selection metric: neg_rmse; CV score -15.6191.

### regression_high: ElasticNet

MAE 3.268; RMSE 4.050; R² 0.242; within ±5 points 82.24%.
Mean-prediction baseline: MAE 3.873; RMSE 4.655; R² -0.002.
Selection metric: neg_rmse; CV score -4.4462.

## Combined workflow

Shortlisted 651 tracks from 14389 holdout tracks.
The high-popularity regressor scores all shortlisted tracks, including false positives. These metrics describe the shortlist, not every track or future Billboard success.
Shortlist regression: {"rows": 651, "mae": 30.89337694783609, "rmse": 42.65376485401698, "r2": -0.9903270662210535, "within_5_percent": 19.969278033794165}.

## Interpretation and limits

A majority-class baseline can have high accuracy while finding zero hits. Use hit precision, recall, F1 and average precision to judge discovery. Regression R² must be compared to a mean baseline. Within ±5 points is a tolerance metric, not classification accuracy.

The original saved RF classifier had 96.70% accuracy on a different random split. The original general-regression Lasso had RMSE 15.542 and R² 0.371; the high-popularity Lasso had RMSE 4.179 and R² 0.223. These are historical notebook outputs, not rerun results. Different samples, stratification, cross-fitted encoding, valid validation and selection criteria mean this run is not a direct reproduction of those numbers.

Random track splitting does not test future releases or independence between artists. Song characteristics alone omit marketing, audience and release timing. The high-popularity model is trained on known hits and may overestimate false positives.

Open `metrics/*_summary.csv` for all models, `predictions/ranked_candidates.csv` for the shortlist, and `executed_notebooks/` for cell-by-cell outputs. Timing and hashes are recorded in `run_manifest.json`.
