# Spotify popularity experiment: results and complete explanation

This document consolidates the completed experiment and the questions discussed about it. It has two parts: one results section, followed by one explanation connecting the whole workflow, its decisions, and their consequences. Measured outcomes, interpretations, and proposed future experiments are distinguished throughout.

## 1. Consolidated results

**The full experiment completed successfully.** It found useful predictive information in the track features, but the combined hit-detection and hit-only scoring approach performed poorly. Its main problem was that the classifier sent many non-hits to a scoring model trained only on actual hits.

| Run detail | Recorded value |
|---|---|
| Run ID | `20261021_141923_142263_full` |
| Status | Completed; artifact verification passed |
| Start | October 21, 2026, 14:19:23 IST |
| Finish | October 21, 2026, 16:25:37 IST |
| Elapsed time | Approximately 2 hours 6 minutes |
| Parallel search workers | 8 |
| Native threads per worker | 1 |
| Raw dataset | 114,000 rows |
| Cleaned dataset used | 71,945 tracks |
| Training partition | 57,556 tracks, including 1,825 hits |
| Held-out test partition | 14,389 tracks, including 456 hits |
| Hit definition | Spotify popularity strictly greater than 70 |
| Cross-validation | Five folds within the training partition |
| Parameter configurations evaluated | 744 across all three modeling experiments |
| Estimator fits, including refits | 3,746 |

The target is **Spotify popularity**, not observed Billboard chart membership. The experiment also does not establish prediction of future popularity from information available before release.

**Selected pipelines and their test results**

“Selected” means chosen using training cross-validation, not chosen because of the best test score.

| Experiment | Selected model | Test population | Main results |
|---|---|---|---|
| Hit/non-hit classification | Gaussian Naive Bayes | All 14,389 test tracks | Precision 25.65%; recall 36.62%; F1 0.3017; accuracy 94.63%; average precision 0.2129; ROC AUC 0.8480 |
| General popularity regression | XGBoost Regressor | All 14,389 test tracks | MAE 11.082; RMSE 15.611; R² 0.3665; 37.10% within ±5 popularity points |
| Hit-only popularity regression | ElasticNet | The 456 actual hits in the test partition | MAE 3.268; RMSE 4.050; R² 0.2417; 82.24% within ±5 points |
| Combined classification → hit-only scoring | Gaussian Naive Bayes → ElasticNet | The 651 tracks predicted to be hits | MAE 30.893; RMSE 42.654; R² −0.9903; 19.97% within ±5 points |

**Classification comparison**

All ten classifier families were trained on both hits and non-hits. CV F1 is the mean hit-class F1 during training cross-validation. The remaining columns describe the held-out test set.

| Classifier | CV F1 | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| All-non-hit baseline | — | 0.00% | 0.00% | 0.0000 | 96.83% |
| **Gaussian Naive Bayes — selected** | **0.3257** | 25.65% | 36.62% | 0.3017 | 94.63% |
| Random Forest | 0.3176 | 25.35% | 39.47% | 0.3087 | 94.40% |
| SVC | 0.2877 | 16.86% | 64.25% | 0.2671 | 88.82% |
| Logistic Regression | 0.2753 | 15.73% | 68.86% | 0.2561 | 87.32% |
| Decision Tree | 0.2491 | 16.31% | 55.70% | 0.2524 | 89.54% |
| MLP neural network | 0.1972 | 26.62% | 17.11% | 0.2083 | 95.88% |
| K-Nearest Neighbors | 0.1758 | 25.93% | 10.75% | 0.1519 | 96.20% |
| Gradient Boosting | 0.1412 | 30.97% | 7.68% | 0.1230 | 96.53% |
| XGBoost | 0.1296 | 38.24% | 8.55% | 0.1398 | 96.66% |
| Bagging | 0.0960 | 31.88% | 4.82% | 0.0838 | 96.66% |

The selected classifier's confusion matrix was:

| Actual class | Predicted non-hit | Predicted hit |
|---|---:|---:|
| Non-hit | 13,449 true negatives | 484 false positives |
| Hit | 289 false negatives | 167 true positives |

The 651-track shortlist therefore contained 167 actual hits and 484 non-hits. Its hit proportion was 25.65%, compared with 3.17% in the test population: approximately eightfold enrichment, but still roughly three incorrect selections for every correct one.

**General regression comparison**

All eight families were trained on the full training partition. Lower CV RMSE was the selection criterion.

| Regressor | CV RMSE | Test MAE | Test RMSE | Test R² | Within ±5 |
|---|---:|---:|---:|---:|---:|
| Training-mean baseline | — | 16.274 | 19.615 | −0.0001 | 16.20% |
| **XGBoost — selected** | **15.6191** | 11.082 | 15.611 | 0.3665 | 37.10% |
| Ridge | 15.6208 | 11.008 | 15.603 | 0.3672 | 38.61% |
| Lasso | 15.6208 | 11.008 | 15.603 | 0.3672 | 38.61% |
| ElasticNet | 15.6208 | 11.008 | 15.603 | 0.3672 | 38.61% |
| Gradient Boosting | 15.6239 | 11.089 | 15.618 | 0.3660 | 37.31% |
| Random Forest | 15.7170 | 11.309 | 15.699 | 0.3594 | 34.78% |
| K-Nearest Neighbors | 16.2046 | 11.954 | 16.222 | 0.3159 | 31.94% |
| AdaBoost | 17.2266 | 13.489 | 17.191 | 0.2318 | 24.44% |

The selected XGBoost pipeline reduced MAE by approximately 31.9% and RMSE by 20.4% against the training-mean baseline. Its CV advantage over the linear models was extremely small; this run does not establish a meaningful superiority over them.

**Hit-only regression comparison**

These same eight regression families were trained separately using only the 1,825 training hits, then evaluated on the 456 actual test hits. No classifier was trained on a hit-only population.

| Regressor | CV RMSE | Test MAE | Test RMSE | Test R² | Within ±5 |
|---|---:|---:|---:|---:|---:|
| Hit-training-mean baseline | — | 3.873 | 4.655 | −0.0019 | 69.52% |
| **ElasticNet — selected** | **4.4462** | 3.268 | 4.050 | 0.2417 | 82.24% |
| Lasso | 4.4463 | 3.265 | 4.047 | 0.2425 | 82.46% |
| Ridge | 4.4470 | 3.259 | 4.044 | 0.2438 | 82.24% |
| Random Forest | 4.5133 | 3.385 | 4.203 | 0.1832 | 81.36% |
| Gradient Boosting | 4.5188 | 3.430 | 4.232 | 0.1718 | 81.58% |
| XGBoost | 4.5284 | 3.430 | 4.240 | 0.1688 | 81.14% |
| AdaBoost | 4.5708 | 3.566 | 4.349 | 0.1254 | 80.04% |
| K-Nearest Neighbors | 4.6271 | 3.465 | 4.357 | 0.1220 | 78.73% |

The linear models were effectively near-tied here too. ElasticNet's MAE improvement against its baseline was approximately 15.6%.

**What happened when classification and hit-only scoring were combined**

The following breakdown was computed from this run's saved shortlist after training; it is diagnostic analysis, not an additional tuning step.

| Shortlisted population | Tracks | Mean actual popularity | Mean predicted popularity | MAE |
|---|---:|---:|---:|---:|
| Actual hits | 167 | 77.95 | 77.56 | 3.66 |
| Non-hits incorrectly shortlisted | 484 | 36.68 | 76.97 | 40.29 |
| All shortlisted tracks | 651 | 47.27 | 77.12 | 30.89 |

The constant hit-training-mean baseline had shortlist MAE 30.56, slightly better than the combined scorer's 30.89. The combined system therefore did not demonstrate a useful score-prediction improvement over that simple reference on its shortlisted population.

**Execution and evidence**

Classification took approximately 85 minutes, general regression 40 minutes, hit-only regression one minute, and the combined inference/evaluation stage two seconds. SVC alone took about 48 minutes. Some MLP fits reached their 500-iteration cap without convergence. The logs also contained worker-stop warnings; the run nevertheless completed and all 744 configurations had finite mean validation scores. The warning messages do not establish their underlying cause.

Primary evidence is available in the run directory:

- [Configuration, status, timestamps and source hashes](runs/20261021_141923_142263_full/run_manifest.json)
- [Cleaning counts](runs/20261021_141923_142263_full/metrics/data_processing.json)
- [Classification comparison](runs/20261021_141923_142263_full/metrics/classification_summary.csv)
- [General regression comparison](runs/20261021_141923_142263_full/metrics/regression_summary.csv)
- [Hit-only regression comparison](runs/20261021_141923_142263_full/metrics/regression_high_summary.csv)
- [Combined metrics](runs/20261021_141923_142263_full/metrics/combined.json)
- [Ranked shortlist with track identities and actual scores](runs/20261021_141923_142263_full/predictions/ranked_candidates.csv)
- [Pipeline progress log](runs/20261021_141923_142263_full/pipeline.log)

The root `VALIDATION.md` documents the earlier smoke test. Its small-sample winners and results must not be confused with this completed full-data run.

## 2. One connected explanation of the experiment and its decisions

**Start with the question, not the algorithm.** The project asks whether a song's measured characteristics can predict its Spotify popularity. It is supervised learning: each example contains input features and an observed answer. The model learns associations between them, much as a house-price model learns from property characteristics and previous sale prices. It does not establish that changing a particular musical characteristic will cause greater popularity.

Spotify popularity ranges from 0 to 100 and reflects listening activity, including its recency. It is not a direct measure of musical quality or the probability of entering a chart. This experiment uses popularity values from a dataset snapshot; it has no explicit future prediction horizon or actual Billboard labels. The original Billboard motivation is therefore broader than what was measured. A defensible presentation describes this as Spotify-popularity prediction and threshold-based hit identification. [Spotify's popularity definition](https://developer.spotify.com/documentation/web-api/reference/get-track)

The experiment translates that broad question into three learning tasks. Classification asks whether popularity exceeds 70. General regression predicts a numerical popularity score across the full population. Hit-only regression predicts numerical scores within the already-popular population. The final combined workflow connects the classifier to the hit-only regressor; the general regressor remains a separate benchmark and does not feed that final workflow.

```text
114,000 raw rows
       ↓ cleaning and deduplication
71,945 retained tracks
       ↓ one saved stratified 80/20 split
57,556 training tracks                  14,389 held-out test tracks
       ↓ training-only CV                          ↓ final evaluation
       ├─ 10 classifier families ──────→ identify predicted hits
       │                                         ↓
       │                                   shortlist of 651 tracks
       │                                         ↓
       ├─ 8 hit-only regressors ───────→ score that shortlist
       │  trained on 1,825 actual hits             ↓
       │                                   ranked candidates
       │
       └─ 8 general regressors ───────→ evaluate scores for all test tracks
```

**The label definition determines what success means.** A hit is a track with popularity strictly above 70. A score of 70 is a non-hit; 71 is a hit. This threshold was inherited from the project and was not established by a business study or sensitivity analysis. It affects how many positive examples exist, which mistakes the classifier makes, and what population enters the hit-only experiment. Changing it would change the task and require a separately validated comparison. It should not be defended as an industry-standard hit definition.

**The model receives track descriptions rather than raw audio.** Its 15 input features are:

| Input | Simple interpretation |
|---|---|
| `duration_ms` | Track length in milliseconds |
| `explicit` | Explicit-content indicator |
| `danceability` | Numerical description of suitability for dancing |
| `energy` | Perceived intensity |
| `key` | Musical pitch class |
| `loudness` | Overall loudness in decibels |
| `mode` | Major/minor indicator |
| `speechiness` | Spoken-word content |
| `acousticness` | Acoustic character |
| `instrumentalness` | Degree of vocal absence |
| `liveness` | Indication of a live setting |
| `valence` | Musical positivity or mood characteristic |
| `tempo` | Estimated beats per minute |
| `time_signature` | Estimated meter |
| `track_genre` | Genre category |

Popularity itself is excluded from the input columns. Track ID, artist, album, and title are also excluded from fitting and retained separately for readable outputs. This limits direct identifier memorization but also omits potentially useful artist-history information. Whether that omission is appropriate depends on the prediction scenario. Artist information available before a prediction date can be legitimate; future audience information would not be.

**Cleaning is a modeling decision because it defines the population.** One missing-value row was removed rather than introducing imputation. Repeated track representations were deduplicated to reduce duplication and related examples crossing evaluation boundaries. The procedure then applied the original project's speechiness, instrumentalness, liveness, and duration rules. The exact effect on row counts was:

| Cleaning decision | Rows remaining | Reason and qualification |
|---|---:|---|
| Remove missing rows | 113,999 | Avoid incomplete inputs; only one row was removed |
| Deduplicate track ID | 89,740 | Reduce repeated track representations |
| Deduplicate artist + album + title | 89,379 | Remove further metadata duplicates |
| Deduplicate artist + title | 81,343 | Reduce repeat versions; may merge legitimate variants |
| Deduplicate album + title | 80,279 | Preserve original procedure; may create collisions across artists |
| Keep speechiness ≤ 0.7 | 79,433 | Restrict strongly spoken content; not a definitive podcast detector |
| Keep instrumentalness ≤ 0.9 | 74,203 | Exclude strongly instrumental recordings; not proof of remakes |
| Keep liveness ≤ 0.8 | 71,945 | Restrict strongly live-sounding recordings; not proof of livestreams |
| Keep nonzero duration | 71,945 | Reject invalid zero durations; no further rows removed |

The thresholds and duplicate rules reproduce the original cleaned CSV values. That establishes consistency, not that the rules are optimal. Some rules can remove valid recordings or discard alternate genre associations. The retained population excludes categories that a real product might need to handle. Their independent effect on predictive quality was not measured because no alternative-cleaning experiment was run.

The raw dataset also has exactly 1,000 rows for each of 114 genres. This is evidence of structured sampling, not a random sample of everything released on Spotify. After cleaning, 113 genre labels remain. Deployment on a different population could have different hit rates, genre proportions, and relationships.

**Rare hits make overall accuracy misleading.** Only 2,281 of 71,945 cleaned tracks are hits, or about 3.17%. A model can obtain approximately 96.8% accuracy simply by calling every track a non-hit. That model misses every actual hit. This is why the experiment records an explicit majority-class baseline and selects classifiers using hit-class F1 instead of overall accuracy.

Precision asks, “Of my selected songs, how many are actually hits?” Recall asks, “Of all actual hits, how many did I find?” F1 combines both. Using the selected classifier's counts:

```text
Precision = true positives / predicted positives
          = 167 / 651 = 25.65%

Recall    = true positives / actual positives
          = 167 / 456 = 36.62%

F1        = 2 × precision × recall / (precision + recall)
          = 0.3017
```

F1 prevents a model from looking useful merely because it rejects many non-hits, but it is still a proxy objective. A team spending money on each selected song might prioritize precision. A discovery team afraid of missing promising songs might prioritize recall. A team able to review exactly 100 songs might care most about precision among the top 100. No business cost function was established here, so the selected operating point is not automatically suitable for deployment.

Some classifiers received class-balancing settings: Logistic Regression, Decision Tree, SVC, and Random Forest. These settings make errors involving rare hits more influential during fitting. Other classifiers use their standard fitting setup, and the selected Gaussian Naive Bayes uses a learned empirical class prior. Thus these results compare the implemented pipelines; they do not establish how every family would perform under identical or comprehensively tuned imbalance handling.

**The split protects evaluation from training.** The saved 80/20 split provides 57,556 training tracks and 14,389 held-out test tracks. Stratification maintains similar hit proportions in both. A fixed seed, 42, makes the random split repeatable; the number has no special scientific meaning. All modeling stages share this partition so a track used for final evaluation cannot have entered another component's training set.

For hit-only regression, the training and test partitions are filtered separately after splitting. Therefore, its 456 test hits never enter its 1,825-row training set. Importantly, these are actual hits selected using known labels for a conditional regression experiment. This does not mean the system knows at prediction time which arbitrary songs really are hits.

The split is by track, not by artist or release date. An additional audit found that 13,055 of the 14,389 test tracks, about 90.7%, have at least one artist represented in training. Although artist IDs are not input features, related tracks may share style and production. This is not direct artist-ID leakage, but it limits the conclusion: performance on held-out tracks from largely familiar artists does not establish performance on unseen artists. Artist-grouped testing and a later-release test period would answer stricter generalization questions. Collaborations complicate group definitions because one track can link multiple artists. [Grouped cross-validation](https://scikit-learn.org/1.8/modules/cross_validation.html#group-k-fold)

**Preprocessing must be learned from the appropriate training data.** Each candidate contains a genre encoder, scaler, PCA transform, and estimator in one saved pipeline. These steps are fitted inside each cross-validation training fold rather than once on the whole dataset. This protects validation and test evaluation from standard preprocessing leakage. At prediction time, each pipeline applies its own fitted transformations to raw features. Sharing a differently fitted classifier's PCA coordinates with a regressor would change the meaning of its inputs.

Genre target encoding turns a text label into a training-derived numerical summary: an estimated hit frequency for classification or an average popularity for regression. Smoothing makes uncertain genre estimates less extreme. This avoids imposing an arbitrary order such as “rock = 1, pop = 2.” Because it uses labels, training encodings are internally cross-fitted: rows are encoded using estimates learned from other training folds. Validation and test encodings use only their relevant training mappings; unseen categories receive the training target mean. This is supervised feature construction, not permission to use a test song's answer. One-hot encoding remains a reasonable untested alternative. [TargetEncoder behavior](https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.TargetEncoder.html)

Standard scaling subtracts a training mean and divides by a training standard deviation. A duration measured in hundreds of thousands should not dominate a distance calculation merely because its units are larger than a danceability value near one. This matters particularly for SVC, nearest neighbors, regularized linear models, neural networks, and PCA. The test partition is transformed with training statistics rather than fitted with its own scaler.

PCA creates new axes that combine the encoded input columns and retains enough components to exceed 95% of input variance. The selected pipelines each retained 13 components from 15 inputs. Their retained variances were about 96.61% for classification, 96.60% for general regression, and 95.79% for hit-only regression. These are percentages of variation in the input representation, not prediction accuracy or percentages of popularity explained. PCA was intended to reduce redundant dimensions, but its benefit was not isolated. It can remove low-variance predictive information, complicate interpretation, and disadvantage trees that prefer splits along original features. A no-PCA comparison is needed before crediting PCA with improvement. [PCA documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.PCA.html)

Other feature representations also deserve testing. Musical key is currently passed through as a number even though pitch classes are circular. Time signature is discrete. The numerical representation is convenient, but it is not proof that ordinary numeric distances are musically appropriate.

**Validation chooses the settings; the test set reports the outcome.** Five-fold cross-validation divides the training partition into five pieces. Each candidate is fitted on four pieces and evaluated on the fifth, rotating through all five. Classification uses stratified folds; regression uses shuffled ordinary folds. The average validation score selects hyperparameters and then the model family. The winning configuration for each family is refitted on its entire training partition. The saved models are not refitted on training plus test data.

A hyperparameter controls how learning occurs: tree depth, number of trees, learning rate, regularization, or number of neighbors. Randomized search samples combinations from configured spaces. The full profile permits up to 30 combinations per family; finite spaces cap some classifiers below 30. It is a full-data experiment, not an exhaustive search or the original much larger search-budget profile. There were 264 classification configurations and 240 in each regression experiment. Five folds plus one refit per family produce 3,746 estimator fits; internal preprocessing fits are additional operations, not additional entries in this estimator-fit count.

For classification, a higher validation F1 is better. For regression, a lower RMSE is better. The search records negative RMSE because its API maximizes scores: −15.619 is better than −15.621. The sign does not indicate a negative prediction error. Once test results have been examined, repeatedly redesigning the model around those results would turn that test set into another validation set; further improvement claims would need fresh evaluation.

**The model-family comparison tests different assumptions.** Logistic Regression learns a relatively simple classification boundary. Lasso, Ridge, and ElasticNet are linear score predictors with regularization. A Decision Tree learns conditional rules. Random Forest combines randomized trees, and Bagging averages models trained with resampling. SVC learns a separating margin and can use nonlinear kernels. Nearest Neighbors uses outcomes of similar examples. Gaussian Naive Bayes models feature distributions within classes. Gradient Boosting and XGBoost build successive trees to correct errors. MLP learns a neural-network mapping. Including these families tests whether simple relationships, local similarity, or more complex interactions are useful under this representation.

The same eight regression families were used in both general and hit-only regression: Lasso, Ridge, ElasticNet, Random Forest Regressor, Gradient Boosting Regressor, AdaBoost Regressor, K-Nearest Neighbors Regressor, and XGBoost Regressor. There was no hit-only classifier: a classifier needs both classes to learn the hit/non-hit distinction in this experiment.

**The selected classifier has useful signal but limited discovery performance.** Gaussian Naive Bayes had mean CV F1 0.3257, narrowly above Random Forest's 0.3176. Their fold standard deviations were approximately 0.0239 and 0.0214. The gap is small relative to fold variation; this is not a formal statistical significance test, and the results do not show a decisive separation. Random Forest actually had slightly higher test F1, but switching winners on that basis would use the test set for selection.

Gaussian Naive Bayes assumes feature independence conditional on class and models each class-conditional feature distribution as Gaussian. Its selected variance-smoothing setting was approximately 0.001668, which stabilizes variance estimates. PCA's global decorrelation does not establish those conditional-independence assumptions. A plausible explanation for the result is that a simple distributional model captured broad patterns without fitting as much noise, but this was not established through ablation or repeated splits. [Naive Bayes assumptions](https://scikit-learn.org/1.8/modules/naive_bayes.html)

The classifier's 94.63% accuracy is lower than the baseline's 96.83%, while its hit discovery is better. Both statements are true. It recovers 167 hits and misses 289. It also creates 484 false alarms. Whether that tradeoff is acceptable depends on the use case. Its ROC AUC of 0.848 indicates useful score ordering, but does not imply a high-precision shortlist at the chosen classification rule. Average precision 0.2129 summarizes precision across recall changes and is different from the 25.65% precision at the current rule. The constant-score baseline's average precision is approximately 0.0317, the test hit prevalence. [Average precision definition](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html)

**General regression demonstrates signal without demonstrating a special need for complexity.** MAE is the average absolute error in popularity points; RMSE penalizes large errors more heavily. XGBoost's MAE of 11.08 means its predictions are off by around eleven points on average. The training-mean baseline always predicts about 35.01 and has MAE 16.27. The improvement indicates useful predictive structure, although large errors remain.

R² compares model squared error with squared error around the mean of the evaluated target values. R² 0.3665 indicates about a 36.7% reduction against that reference, not “36.7% correct predictions.” The baseline is fitted using the training mean, so its test R² can be slightly below zero. The ±5 measure counts predictions within five popularity points and is a tolerance percentage rather than classification accuracy. MAPE was omitted because zero popularity scores make percentage errors unsuitable.

XGBoost's winning CV RMSE was only about 0.0018 points below Ridge's, against fold variability around 0.104. Linear models had slightly better held-out MAE and RMSE. The selection rule chose XGBoost correctly, but the experiment does not demonstrate a meaningful gain from its complexity. Its selected configuration used 200 trees, depth 5, learning rate 0.05, row subsampling 0.7, column subsampling 0.6, `reg_lambda=1.5`, `reg_alpha=0`, and `gamma=0.4`. These settings control model capacity and fitting; their individual contributions were not separately measured.

**The hit-only model is a specialist that assumes its input belongs to the high-popularity group.** Suppose the dataset contains songs with popularity 20, 55, 74, 82, and 95. Only the songs scoring 74, 82, and 95 enter hit-only training. The model learns how to distinguish moderately popular hits from more popular hits. It does not learn how songs scoring 20 or 55 behave. Its question is, “Given that this song is a hit, what popularity score should it receive?” The classifier's different question is, “Is this song likely to be a hit at all?”

An analogy is a teacher who assesses only advanced students. The teacher learns differences within that group, but has little experience evaluating beginners. Sending a beginner to that specialist does not turn the beginner into an advanced student; it violates the assumption behind the specialist's training population.

ElasticNet was selected for hit-only regression. It combines L1 and L2 regularization: one can push some coefficients to zero, while the other shrinks coefficients. Here the coefficients refer to PCA components. The selected settings were `alpha≈0.0351`, `l1_ratio=0.35`, `max_iter=1000`, and `tol=0.0001`. Its validation advantage over Lasso and Ridge was tiny. The hit-only training-mean baseline predicts about 76.34, and even that baseline is within five points for 69.52% of actual test hits. The model's 82.24% within-five result therefore must be interpreted alongside R² 0.2417 and its restricted target range. It is not 82% accuracy for arbitrary songs.

**The combined model's failure follows from a population mismatch.** Gaussian Naive Bayes first predicts which test songs are hits. ElasticNet then scores those predicted hits, receiving raw features and applying its own saved preprocessing. The label used to admit a song to training was “actual popularity > 70,” but the rule admitting a song at inference is “classifier predicted hit.” Those rules are not equivalent when the classifier makes mistakes.

For a simple illustration, a song with actual popularity 25 can be incorrectly labeled a hit and then receive a specialist's estimate of 77, producing a 52-point error. The regressor is not explicitly constrained to output above 70; it tends toward high scores because it learned only from high-scoring examples. This is not repaired merely by increasing the search budget or adding CPU cores.

In this run, the specialist's MAE was 3.66 on the 167 actual hits in the shortlist, but 40.29 on the 484 false positives. It predicted about 76.97 on average for false positives whose actual mean was 36.68. That mismatch dominates the combined MAE of 30.89. Its negative R² means it is worse in squared-error terms than a reference predicting the evaluated shortlist's own mean. It does not mean negative song popularity.

The combined regression metrics describe only the 651 shortlisted tracks. Rejected tracks receive no score from this combined stage, and the 289 missed hits cannot be recovered by a later ranking model. It would be misleading to compare the combined 30.89-point MAE with the general regressor's 11.08-point MAE without acknowledging their different evaluation populations.

A post-analysis comparison on the same 651 shortlisted tracks gives the general XGBoost regressor MAE 24.05, lower than the specialist's 30.89 but still large. The existing combined ranking's top 100 entries contain 38 actual hits. These diagnostics suggest that the specialist is poorly matched to its inputs; they are not independently validated decisions to replace or tune it using this test set.

**A completed workflow is not the same as a deployment-ready prediction system.** The runner executes five corrected notebook copies sequentially, stores output in an isolated run directory, and saves each encoder/scaler/PCA/model together. This corrects the original workflow's inconsistent model paths, incompatible transform handoff, and training-on-training parameter evaluation. Original root notebooks and their saved historical outputs were preserved; corrected automation is under `pipeline_notebooks/`. The original “overfitted regression” description is not a goal of the corrected experiment: high training fit is not useful if held-out predictions deteriorate.

The old and corrected results are not a controlled before/after comparison. They differ in split, target-encoding procedure, validation, some search choices, imbalance treatment, and classifier selection metric. It is therefore not valid to assign a change in score to a single correction. Likewise, the current run has no feature-attribution study, so it cannot establish that danceability, energy, or another particular field is the primary reason for popularity.

**The execution questions do not change the statistical task.** The original small test took around 38 seconds with 8,000 tracks and smaller searches, but full-run time could not be reliably inferred by multiplying that duration: SVC and larger ensembles scale differently. The earlier estimate of several hours was rough. The measured eight-worker full run took about two hours six minutes. That is the observed runtime for this configuration, not a guarantee for another run or machine.

The project uses `.venv` to isolate Python dependencies. Activating it with `source .venv/bin/activate` makes `python` refer to that environment; using `.venv/bin/python` selects it explicitly without activation. `caffeinate -i` prevents idle sleep while its command runs. It does not make model fitting faster. Ctrl+C interrupts the run. Restarting creates a new results folder and starts from the beginning because this runner does not implement resumption, although earlier artifacts remain saved.

The visible message “classification: cell 1 finished” marks completion of the notebook's setup cell. The following cell runs all ten classifier searches, so terminal cell-completion messages can remain unchanged for a long time while fitting continues. `pipeline.log` reports model starts and finishes. The relevant stage's kernel log records individual completed cross-validation fits. Continued log updates and CPU activity indicate work; silence alone does not establish a hang because a fit can produce no output until it finishes. Stale logs together with near-zero CPU activity warrant investigation, but are not proof of a hang.

The actual run used eight parallel search workers, with one native thread per worker. These workers handle candidate/fold fits concurrently; the five notebooks and model families are not all launched together. The user reported an M4 Pro with 48 GB RAM and 12 cores. More workers can help independent fits, but twelve workers do not guarantee a twelvefold speedup: fits share memory bandwidth, some stages are serial, and concurrency can increase resource pressure. This experiment did not benchmark two versus eight versus twelve workers under controlled conditions. Changing `--n-jobs` requires restarting the current runner.

**What an ML engineer should hear about the limits.** Threshold 70 is a project convention, cleaning rules are inherited heuristics, target encoding and PCA are choices without ablations, and model rankings depend on the configured search spaces. Some algorithms were not fully converged or comprehensively tuned. The raw genre-balanced sampling and artist overlap limit deployment claims. No temporal experiment established future prediction, and no Billboard labels were used. Predicted hit scores were not checked for probability calibration, so a score of 0.8 should not automatically be called an 80% chance of success. [Probability calibration](https://scikit-learn.org/1.8/modules/calibration.html)

**How to approach hit prediction from scratch after learning from this experiment.** This is proposed future work, not something performed in the recorded run. First define the outcome and prediction time: for example, use information available seven days after release to predict entry into a specified country's chart during the following eight weeks. A pre-release prediction would require a different feature set. Historical outcomes should then be collected for successful and unsuccessful releases, with every feature restricted to information available by the prediction date.

Useful candidate information includes audio characteristics, artist history measured before prediction, release context, and early audience response if the prediction time allows it. Future stream totals or a later popularity score cannot be used to predict an earlier outcome. Train on older releases, choose settings on a later validation period, and test on the latest period. Add an unseen-artist evaluation if discovering new artists is a goal. Start with prevalence, artist-history, and simple logistic-regression baselines before adding a boosting model.

The first deliverable should be one classifier that scores all eligible songs and produces a ranked list. If a team reviews 100 songs per week, evaluate how many of its top 100 become hits and how many total hits it misses. If probabilities are used for decisions, assess calibration. A separate hit-only regressor should not be added automatically: the classifier already supplies a ranking score, and any second stage must demonstrate an improvement in the actual decision objective. If numerical popularity forecasting is needed, evaluate that task explicitly on the population the scorer will actually receive.

For the existing dataset, useful next experiments would include a genre-only baseline, audio-only versus audio-plus-genre comparisons, one-hot versus target encoding, no-PCA pipelines, and validation-based decision-threshold selection. These would clarify which choices contribute useful signal. Repeated splits or appropriate uncertainty analysis would help assess the near-ties between models. Any design changes informed by the current test findings need fresh evaluation rather than another claim of improvement on the same inspected test set.

**A concise explanation to give an ML engineer:**

> We built a reproducible tabular experiment to predict Spotify popularity from track characteristics. Cleaning left 71,945 tracks, and a shared stratified 80/20 split kept evaluation rows out of training. We compared ten classifiers and eight regressors in two regression settings using training-only five-fold cross-validation. Each pipeline included cross-fitted genre encoding, scaling, PCA, and its estimator.
>
> Gaussian Naive Bayes narrowly led classification validation F1; on held-out tracks it found 167 of 456 hits, with 25.65% precision and 36.62% recall. General popularity regression achieved MAE 11.08 and R² 0.367, with XGBoost essentially tied with regularized linear models. ElasticNet scored actual hits with MAE 3.27, but the combined workflow had MAE 30.89 because 484 of its 651 shortlisted tracks were non-hits outside the specialist's training population.
>
> The run demonstrates useful predictive signal and exposes a weakness in the two-stage design. It does not establish future Billboard prediction, unseen-artist performance, a decisive advantage for complex models, or deployment readiness. The next work should align the outcome, prediction date, evaluation split, and ranking objective with the intended use case.
