"""Run-scoped data preparation, honest evaluation and model hand-off for notebooks."""

import hashlib
import json
import logging
import os
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    KFold, ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split,
)
from joblib import parallel_config
from threadpoolctl import threadpool_limits

from pipeline_models import FEATURES, make_pipeline, model_specs


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunContext:
    def __init__(self, root, run_dir, config, stage):
        self.root, self.run_dir = Path(root), Path(run_dir)
        self.config, self.stage = config, stage
        self.logger = logging.getLogger(f"spotify.{stage}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
        for destination in [self.run_dir / "pipeline.log", self.path("logs", f"{stage}.log")]:
            handler = logging.FileHandler(destination)
            handler.setFormatter(logging.Formatter(f"%(asctime)s [{stage}] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())

    @classmethod
    def from_environment(cls, stage):
        run_dir = os.environ.get("SPOTIFY_RUN_DIR")
        if not run_dir:
            raise RuntimeError(
                "Start notebooks with .venv/bin/python run_pipeline.py --profile smoke. "
                "The runner supplies the run directory, configuration and correct Python kernel."
            )
        config = json.loads((Path(run_dir) / "config.json").read_text())
        return cls(Path(__file__).parent, run_dir, config, stage)

    def path(self, folder, filename):
        path = self.run_dir / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, message, *args):
        self.logger.info(message, *args)

    def data(self):
        frame = pd.read_csv(self.path("data", "modeling_dataset.csv")).set_index("row_id")
        split = pd.read_csv(self.path("data", "split.csv")).set_index("row_id")
        if not frame.index.is_unique or not split.index.is_unique or set(frame.index) != set(split.index):
            raise ValueError("Dataset and saved split row IDs do not match uniquely")
        if set(split["split"]) != {"train", "test"}:
            raise ValueError("Saved split must contain only nonempty train and test partitions")
        train = frame.loc[split.index[split["split"] == "train"]]
        test = frame.loc[split.index[split["split"] == "test"]]
        if set(train.index) & set(test.index):
            raise ValueError("Training and holdout rows overlap")
        return train, test


def prepare_data(ctx):
    source = ctx.root / "spotify_tracks_dataset.csv"
    df = pd.read_csv(source)
    counts = {"raw_rows": len(df)}

    def record(label):
        counts[label] = len(df)
        ctx.log("%s: %d rows", label, len(df))

    df = df.dropna().drop(columns=[c for c in df if c.startswith("Unnamed:")])
    record("after_missing_values")
    for keys in [["track_id"], ["artists", "album_name", "track_name"],
                 ["artists", "track_name"], ["album_name", "track_name"]]:
        df = df.drop_duplicates(subset=keys)
        record("after_dedup_" + "_".join(keys))
    df["explicit"] = df["explicit"].astype("int64")
    # These are the original notebook's effective thresholds, including 0.9
    # for instrumentalness (the old README said 0.8).
    for name, mask in [
        ("speechiness", lambda x: x.speechiness <= 0.7),
        ("instrumentalness", lambda x: x.instrumentalness <= 0.9),
        ("liveness", lambda x: x.liveness <= 0.8),
        ("nonzero_duration", lambda x: x.duration_ms != 0),
    ]:
        df = df.loc[mask(df)]
        record("after_" + name)
    df.index.name = "row_id"
    metadata = df[["track_id", "artists", "album_name", "track_name"]].copy()
    cleaned = df[FEATURES + ["popularity"]].copy()
    cleaned.to_csv(ctx.path("data", "processed_dataset.csv"))
    metadata.to_csv(ctx.path("data", "track_metadata.csv"))
    threshold = ctx.config["hit_threshold"]
    sample_rows = ctx.config["sample_rows"]
    if sample_rows is not None and sample_rows < len(cleaned):
        ids, _ = train_test_split(
            cleaned.index, train_size=sample_rows,
            stratify=cleaned.popularity > threshold, random_state=ctx.config["seed"],
        )
        modeling = cleaned.loc[sorted(ids)]
    else:
        modeling = cleaned.copy()
    train_ids, test_ids = train_test_split(
        modeling.index, test_size=ctx.config["test_size"],
        stratify=modeling.popularity > threshold, random_state=ctx.config["seed"],
    )
    split = pd.DataFrame({"row_id": modeling.index, "split": "train"}).set_index("row_id")
    split.loc[test_ids, "split"] = "test"
    modeling.to_csv(ctx.path("data", "modeling_dataset.csv"))
    split.to_csv(ctx.path("data", "split.csv"))
    counts.update({
        "cleaned_rows": len(cleaned), "modeling_rows": len(modeling),
        "train_rows": len(train_ids), "test_rows": len(test_ids),
        "modeling_hits": int((modeling.popularity > threshold).sum()),
        "train_hits": int((modeling.loc[train_ids].popularity > threshold).sum()),
        "test_hits": int((modeling.loc[test_ids].popularity > threshold).sum()),
        "source_sha256": sha256(source),
        "split_sha256": sha256(ctx.path("data", "split.csv")),
    })
    legacy = pd.read_csv(ctx.root / "spotify_tracks_dataset_processed.csv")
    try:
        pd.testing.assert_frame_equal(
            cleaned.reset_index(drop=True)[legacy.columns.drop("Unnamed: 0")],
            legacy.drop(columns="Unnamed: 0"), check_dtype=False,
        )
        counts["matches_original_processed_values"] = True
    except AssertionError:
        counts["matches_original_processed_values"] = False
    save_json(ctx.path("metrics", "data_processing.json"), counts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cleaned.popularity.hist(bins=25, ax=axes[0])
    axes[0].set(title="Cleaned Spotify popularity", xlabel="Popularity", ylabel="Track count")
    (modeling.popularity > threshold).value_counts().reindex([False, True]).plot.bar(ax=axes[1])
    axes[1].set(title="Modeling class balance", xticklabels=["Non-hit", "Hit"], ylabel="Track count")
    fig.tight_layout()
    fig.savefig(ctx.path("plots", "data_distribution.png"))
    plt.close(fig)
    ctx.log("Prepared %d modeling rows; %d train / %d holdout; hits %d / %d",
            len(modeling), len(train_ids), len(test_ids), counts["train_hits"], counts["test_hits"])
    return pd.Series(counts, name="value").to_frame()


def classification_metrics(y, prediction, scores=None):
    result = {
        "rows": len(y), "actual_hits": int(np.sum(y)),
        "predicted_hits": int(np.sum(prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "hit_precision": float(precision_score(y, prediction, zero_division=0)),
        "hit_recall": float(recall_score(y, prediction, zero_division=0)),
        "hit_f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    if scores is not None and len(np.unique(y)) == 2:
        result.update({"average_precision": float(average_precision_score(y, scores)),
                       "roc_auc": float(roc_auc_score(y, scores))})
    return result


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if len(y) == 0:
        return {"rows": 0, "mae": None, "rmse": None, "r2": None, "within_5_percent": None}
    return {
        "rows": len(y), "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else None,
        "within_5_percent": float(np.mean(np.abs(y - prediction) <= 5) * 100),
    }


def classifier_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_models(ctx, classification=False, high_only=False):
    train, test = ctx.data()
    threshold = ctx.config["hit_threshold"]
    if high_only:
        train = train.loc[train.popularity > threshold]
        test = test.loc[test.popularity > threshold]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train = (train.popularity > threshold).astype(int) if classification else train.popularity
    y_test = (test.popularity > threshold).astype(int) if classification else test.popularity
    folds, seed = ctx.config["cv_folds"], ctx.config["seed"]
    if len(train) < folds * 10 or len(test) < 2:
        raise ValueError(f"Too few rows for {ctx.stage}; increase sample_rows")
    if classification and y_train.value_counts().min() < folds * 5:
        raise ValueError("Too few hits for nested genre encoding; increase sample_rows")
    cv = (StratifiedKFold(folds, shuffle=True, random_state=seed) if classification
          else KFold(folds, shuffle=True, random_state=seed))
    scoring = ({"hit_f1": "f1", "average_precision": "average_precision",
                "balanced_accuracy": "balanced_accuracy"} if classification
               else {"neg_rmse": "neg_root_mean_squared_error"})
    refit = "hit_f1" if classification else "neg_rmse"
    budget = ctx.config["classification_iterations" if classification else
                        "high_regression_iterations" if high_only else "regression_iterations"]
    baseline = DummyClassifier(strategy="most_frequent") if classification else DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    metric = classification_metrics if classification else regression_metrics
    base_pred = baseline.predict(X_test)
    baseline_metrics = metric(y_test, base_pred, baseline.predict_proba(X_test)[:, 1]) if classification else metric(y_test, base_pred)
    rows = [{"model": "MajorityClassBaseline" if classification else "MeanBaseline",
             "cv_selection_score": None,
             **{f"test_{k}": v for k, v in baseline_metrics.items() if not isinstance(v, list)}}]
    save_json(ctx.path("metrics", f"{ctx.stage}_baseline.json"), baseline_metrics)
    joblib.dump(baseline, ctx.path("models", f"{ctx.stage}_baseline.joblib"))
    best_score, best_name, best_estimator = -np.inf, None, None
    for name, (estimator, params) in model_specs(classification, ctx.config).items():
        started = time.monotonic()
        pipe = make_pipeline(estimator, classification, seed)
        grid = {f"model__{key}": value for key, value in params.items()}
        n_iter = min(budget, len(ParameterGrid(grid)))
        ctx.log("START %s: %d candidates x %d folds + refit; train=%d holdout=%d",
                name, n_iter, folds, len(train), len(test))
        search = RandomizedSearchCV(
            pipe, grid, n_iter=n_iter, scoring=scoring, refit=refit, cv=cv,
            random_state=seed, n_jobs=ctx.config["n_jobs"],
            pre_dispatch=ctx.config["n_jobs"], error_score="raise", verbose=2,
        )
        with parallel_config(backend="loky", inner_max_num_threads=ctx.config["native_threads"]):
            with threadpool_limits(limits=ctx.config["native_threads"]):
                search.fit(X_train, y_train)
                best = search.best_estimator_
                prediction = best.predict(X_test)
                train_prediction = best.predict(X_train)
                if classification:
                    test_metrics = metric(y_test, prediction, classifier_scores(best, X_test))
                    train_metrics = metric(y_train, train_prediction, classifier_scores(best, X_train))
                else:
                    test_metrics, train_metrics = metric(y_test, prediction), metric(y_train, train_prediction)
        model_path = ctx.path("models", f"{ctx.stage}_{name}.joblib")
        joblib.dump(best, model_path)
        # Verify the persisted pipeline reproduces predictions from raw features.
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(reloaded.predict(X_test.iloc[:10]), prediction[:10], rtol=1e-6)
        pd.DataFrame(search.cv_results_).to_csv(ctx.path("cv_results", f"{ctx.stage}_{name}.csv"), index=False)
        elapsed = time.monotonic() - started
        details = {"model": name, "best_parameters": search.best_params_,
                   "cv_selection_metric": refit, "cv_selection_score": float(search.best_score_),
                   "train": train_metrics, "test": test_metrics,
                   "seconds": elapsed, "candidates": n_iter,
                   "pca_components": int(best.named_steps["pca"].n_components_)}
        save_json(ctx.path("metrics", f"{ctx.stage}_{name}.json"), details)
        rows.append({"model": name, "cv_selection_score": float(search.best_score_), "seconds": elapsed,
                     **{f"test_{k}": v for k, v in test_metrics.items() if not isinstance(v, list)},
                     **{f"train_{k}": v for k, v in train_metrics.items() if not isinstance(v, list)}})
        pd.DataFrame(rows).to_csv(ctx.path("metrics", f"{ctx.stage}_summary.csv"), index=False)
        predictions = pd.DataFrame({"row_id": test.index, "actual": y_test, "prediction": prediction})
        if classification:
            predictions["hit_score"] = classifier_scores(best, X_test)
        predictions.to_csv(ctx.path("predictions", f"{ctx.stage}_{name}.csv"), index=False)
        if search.best_score_ > best_score:
            best_score, best_name, best_estimator = float(search.best_score_), name, best
        ctx.log("DONE %s: CV %s=%.4f; holdout %s; %.1fs", name, refit, search.best_score_, test_metrics, elapsed)
    joblib.dump(best_estimator, ctx.path("models", f"{ctx.stage}_best.joblib"))
    selection = {"model": best_name, "metric": refit, "cv_score": best_score,
                 "selected_using": "training cross-validation only", "train_rows": len(train),
                 "holdout_rows": len(test), "baseline": baseline_metrics}
    save_json(ctx.path("metrics", f"{ctx.stage}_selection.json"), selection)
    if classification:
        ConfusionMatrixDisplay.from_predictions(
            y_test, best_estimator.predict(X_test), labels=[0, 1], display_labels=["Non-hit", "Hit"],
        )
        plt.title(f"Holdout confusion matrix: {best_name}")
        plt.tight_layout()
        plt.savefig(ctx.path("plots", "classification_confusion_matrix.png"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, best_estimator.predict(X_test), alpha=0.25, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set(xlabel="Actual popularity", ylabel="Predicted popularity", title=f"{ctx.stage}: {best_name}")
        fig.tight_layout()
        fig.savefig(ctx.path("plots", f"{ctx.stage}_predictions.png"))
        plt.close(fig)
    ctx.log("Selected %s using %s from training folds only", best_name, refit)
    return pd.DataFrame(rows)


def run_combined(ctx):
    _, test = ctx.data()
    classifier = joblib.load(ctx.path("models", "classification_best.joblib"))
    regressor = joblib.load(ctx.path("models", "regression_high_best.joblib"))
    baseline = joblib.load(ctx.path("models", "regression_high_baseline.joblib"))
    # Each pipeline receives original features and applies its OWN fitted transforms.
    prediction = classifier.predict(test[FEATURES])
    hit_scores = classifier_scores(classifier, test[FEATURES])
    selected = test.loc[prediction == 1].copy()
    metrics = classification_metrics(
        (test.popularity > ctx.config["hit_threshold"]).astype(int), prediction, hit_scores,
    )
    output = pd.DataFrame({"row_id": selected.index, "actual_popularity": selected.popularity})
    if len(selected):
        score_prediction = regressor.predict(selected[FEATURES])
        output["predicted_popularity"] = score_prediction
        output["hit_score"] = hit_scores[prediction == 1]
        metrics["shortlist_regression"] = regression_metrics(selected.popularity, score_prediction)
        metrics["shortlist_mean_baseline"] = regression_metrics(
            selected.popularity, baseline.predict(selected[FEATURES]),
        )
        output = output.sort_values("predicted_popularity", ascending=False)
    else:
        output["predicted_popularity"] = pd.Series(dtype=float)
        output["hit_score"] = pd.Series(dtype=float)
        metrics["shortlist_regression"] = regression_metrics([], [])
        metrics["shortlist_mean_baseline"] = regression_metrics([], [])
    metadata = pd.read_csv(ctx.path("data", "track_metadata.csv"))
    output = output.reset_index(drop=True).merge(metadata, on="row_id", how="left", validate="one_to_one")
    output.to_csv(ctx.path("predictions", "ranked_candidates.csv"), index=False)
    save_json(ctx.path("metrics", "combined.json"), metrics)
    ctx.log("Combined holdout: %d shortlisted / %d; metrics %s", len(selected), len(test), metrics)
    return output.head(20)


def write_report(ctx):
    """Generate a short, run-specific report from actual saved measurements."""
    data = json.loads(ctx.path("metrics", "data_processing.json").read_text())
    combined = json.loads(ctx.path("metrics", "combined.json").read_text())
    lines = [
        f"# Spotify results: {ctx.config['profile']}", "",
        "This run predicts Spotify popularity, not measured Billboard chart membership.", "",
        f"Raw rows: {data['raw_rows']:,}; cleaned: {data['cleaned_rows']:,}; modeled: {data['modeling_rows']:,}.",
        f"Training rows: {data['train_rows']:,}; holdout rows: {data['test_rows']:,} "
        f"({data['test_hits']} actual hits). Hit means popularity > {ctx.config['hit_threshold']}.", "",
        "All stages share a saved stratified holdout. Every estimator stores its own genre encoder, scaler, "
        "PCA and model. Preprocessing is fitted within training folds; genre training encodings are cross-fitted.",
        "Winners are selected using cross-validation, never holdout scores. "
        "Training metrics use the final refitted encoder and are descriptive only.", "",
        "## Measured results", "",
    ]
    for stage in ["classification", "regression", "regression_high"]:
        selection = json.loads(ctx.path("metrics", f"{stage}_selection.json").read_text())
        result = json.loads(ctx.path("metrics", f"{stage}_{selection['model']}.json").read_text())
        metrics, base = result["test"], selection["baseline"]
        lines += [f"### {stage}: {selection['model']}", ""]
        if stage == "classification":
            lines += [
                f"Accuracy {metrics['accuracy']:.2%}; majority baseline {base['accuracy']:.2%}.",
                f"Hit precision {metrics['hit_precision']:.2%}; recall {metrics['hit_recall']:.2%}; "
                f"F1 {metrics['hit_f1']:.4f}; average precision {metrics['average_precision']:.4f}.",
                f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}.",
            ]
        else:
            lines += [
                f"MAE {metrics['mae']:.3f}; RMSE {metrics['rmse']:.3f}; R² {metrics['r2']:.3f}; "
                f"within ±5 points {metrics['within_5_percent']:.2f}%.",
                f"Mean-prediction baseline: MAE {base['mae']:.3f}; RMSE {base['rmse']:.3f}; R² {base['r2']:.3f}.",
            ]
        lines += [f"Selection metric: {selection['metric']}; CV score {selection['cv_score']:.4f}.", ""]
    shortlist = combined["shortlist_regression"]
    lines += [
        "## Combined workflow", "",
        f"Shortlisted {combined['predicted_hits']} tracks from {combined['rows']} holdout tracks.",
        "The high-popularity regressor scores all shortlisted tracks, including false positives. "
        "These metrics describe the shortlist, not every track or future Billboard success.",
        f"Shortlist regression: {json.dumps(shortlist)}.", "",
        "## Interpretation and limits", "",
        "A majority-class baseline can have high accuracy while finding zero hits. Use hit precision, recall, "
        "F1 and average precision to judge discovery. Regression R² must be compared to a mean baseline. "
        "Within ±5 points is a tolerance metric, not classification accuracy.", "",
        "The original saved RF classifier had 96.70% accuracy on a different random split. "
        "The original general-regression Lasso had RMSE 15.542 and R² 0.371; the high-popularity Lasso "
        "had RMSE 4.179 and R² 0.223. These are historical notebook outputs, not rerun results. "
        "Different samples, stratification, cross-fitted encoding, valid validation and selection criteria "
        "mean this run is not a direct reproduction of those numbers.", "",
        "Random track splitting does not test future releases or independence between artists. "
        "Song characteristics alone omit marketing, audience and release timing. "
        "The high-popularity model is trained on known hits and may overestimate false positives.", "",
    ]
    if ctx.config["sample_rows"] is not None:
        lines += ["This is a small-sample pipeline verification. Its model rankings and metrics are unstable; "
                  "use the full-data run for the next experiment.", ""]
    lines += ["Open `metrics/*_summary.csv` for all models, `predictions/ranked_candidates.csv` for the shortlist, "
              "and `executed_notebooks/` for cell-by-cell outputs. Timing and hashes are recorded in `run_manifest.json`.", ""]
    (ctx.run_dir / "RESULTS.md").write_text("\n".join(lines))
    return ctx.run_dir / "RESULTS.md"
