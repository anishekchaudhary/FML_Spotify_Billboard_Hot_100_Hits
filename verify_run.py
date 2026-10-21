#!/usr/bin/env python3
"""Verify that a completed run contains successful notebooks and consistent results."""

import argparse
import json
from pathlib import Path

import nbformat
import pandas as pd

from run_pipeline import STAGES


def verify(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed", manifest["status"]
    assert len(manifest["stages"]) == 5
    assert all(stage["status"] == "completed" for stage in manifest["stages"])
    for _, name in STAGES:
        notebook = nbformat.read(run_dir / "executed_notebooks" / Path(name).name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, name
                assert not any(output.output_type == "error" for output in cell.outputs), name
    data = pd.read_csv(run_dir / "data/modeling_dataset.csv")
    split = pd.read_csv(run_dir / "data/split.csv")
    assert data.row_id.is_unique and split.row_id.is_unique
    assert set(data.row_id) == set(split.row_id)
    train_ids = set(split.loc[split.split == "train", "row_id"])
    test_ids = set(split.loc[split.split == "test", "row_id"])
    assert not train_ids & test_ids
    config = manifest["config"]
    for stage in ["classification", "regression", "regression_high"]:
        family_key = "classifier_models" if stage == "classification" else "regressor_models"
        summary = pd.read_csv(run_dir / "metrics" / f"{stage}_summary.csv")
        assert len(summary) == len(config[family_key]) + 1, stage
        assert set(config[family_key]) <= set(summary.model), stage
        selection = json.loads((run_dir / "metrics" / f"{stage}_selection.json").read_text())
        assert selection["selected_using"] == "training cross-validation only"
        for name in config[family_key]:
            predictions = pd.read_csv(run_dir / "predictions" / f"{stage}_{name}.csv")
            expected_ids = test_ids
            if stage == "regression_high":
                expected_ids = set(data.loc[data.row_id.isin(test_ids) &
                                           (data.popularity > config["hit_threshold"]), "row_id"])
            assert set(predictions.row_id) == expected_ids, (stage, name)
            assert predictions.prediction.notna().all()
            assert (run_dir / "models" / f"{stage}_{name}.joblib").exists()
        assert (run_dir / "models" / f"{stage}_best.joblib").exists()
    combined = json.loads((run_dir / "metrics/combined.json").read_text())
    ranked = pd.read_csv(run_dir / "predictions/ranked_candidates.csv")
    assert set(ranked.row_id) <= test_ids
    assert len(ranked) == combined["predicted_hits"] == combined["shortlist_regression"]["rows"]
    assert ranked.predicted_popularity.is_monotonic_decreasing
    assert (run_dir / "RESULTS.md").exists()
    print(f"PASS: five notebooks; all model families; isolated holdout; consistent shortlist.\n{run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    verify(parser.parse_args().run_dir)
