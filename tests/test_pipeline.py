"""Checks for shared holdouts and the combined stage's evaluation boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from pipeline_lib import RunContext, regression_metrics, run_combined
from pipeline_models import FEATURES
from run_pipeline import ROOT, validate_config


class PipelineBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = json.loads((ROOT / "configs/smoke.json").read_text())
        self.run_dir = Path(self.temporary.name)
        self.ctx = RunContext(ROOT, self.run_dir, self.config, "test")
        self.addCleanup(self.close_logs)
        data = pd.DataFrame({name: [0.2] * 6 for name in FEATURES})
        data["track_genre"] = "pop"
        data["popularity"] = [10, 20, 80, 90, 30, 75]
        data.index = pd.Index([11, 12, 13, 14, 15, 16], name="row_id")
        data.to_csv(self.ctx.path("data", "modeling_dataset.csv"))
        split = pd.DataFrame({"split": ["train"] * 4 + ["test"] * 2}, index=data.index)
        split.to_csv(self.ctx.path("data", "split.csv"))
        metadata = pd.DataFrame({"track_id": [f"song-{i}" for i in data.index],
                                 "artists": "artist", "album_name": "album", "track_name": "song"}, index=data.index)
        metadata.to_csv(self.ctx.path("data", "track_metadata.csv"))
        self.train, self.test = data.iloc[:4], data.iloc[4:]
        baseline = DummyRegressor(strategy="mean").fit(self.train[FEATURES], self.train.popularity)
        joblib.dump(baseline, self.ctx.path("models", "regression_high_baseline.joblib"))
        regressor = DummyRegressor(strategy="constant", constant=80).fit(self.train[FEATURES], self.train.popularity)
        joblib.dump(regressor, self.ctx.path("models", "regression_high_best.joblib"))

    def close_logs(self):
        for handler in list(self.ctx.logger.handlers):
            handler.close()
            self.ctx.logger.removeHandler(handler)

    def classifier(self, constant):
        model = DummyClassifier(strategy="constant", constant=constant).fit(self.train[FEATURES], [0, 0, 1, 1])
        joblib.dump(model, self.ctx.path("models", "classification_best.joblib"))

    def test_saved_split_is_shared_and_disjoint(self):
        train, test = self.ctx.data()
        self.assertEqual(list(train.index), [11, 12, 13, 14])
        self.assertEqual(list(test.index), [15, 16])
        self.assertFalse(set(train.index) & set(test.index))

    def test_duplicate_split_rows_are_rejected(self):
        path = self.ctx.path("data", "split.csv")
        split = pd.read_csv(path)
        pd.concat([split, split.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "uniquely"):
            self.ctx.data()

    def test_empty_shortlist_is_a_valid_result(self):
        self.classifier(0)
        result = run_combined(self.ctx)
        metrics = json.loads(self.ctx.path("metrics", "combined.json").read_text())
        self.assertTrue(result.empty)
        self.assertEqual(metrics["shortlist_regression"]["rows"], 0)
        self.assertIsNone(metrics["shortlist_regression"]["rmse"])
        self.assertEqual(metrics["hit_recall"], 0)

    def test_unknown_partition_is_rejected(self):
        path = self.ctx.path("data", "split.csv")
        split = pd.read_csv(path)
        split.loc[0, "split"] = "validation_typo"
        split.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "partitions"):
            self.ctx.data()

    def test_false_positives_remain_in_shortlist_evaluation(self):
        self.classifier(1)
        result = run_combined(self.ctx)
        metrics = json.loads(self.ctx.path("metrics", "combined.json").read_text())
        self.assertEqual(set(result.row_id), {15, 16})
        self.assertEqual(metrics["shortlist_regression"]["rows"], 2)
        self.assertAlmostEqual(metrics["shortlist_regression"]["mae"], (50 + 5) / 2)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1], [0, 1]])

    def test_single_row_regression_does_not_emit_nan_r2(self):
        result = regression_metrics(np.array([80]), np.array([78]))
        self.assertIsNone(result["r2"])
        self.assertEqual(result["within_5_percent"], 100)
        json.dumps(result, allow_nan=False)

    def test_zero_worker_configuration_is_rejected(self):
        self.config["n_jobs"] = 0
        with self.assertRaisesRegex(ValueError, "n_jobs"):
            validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
