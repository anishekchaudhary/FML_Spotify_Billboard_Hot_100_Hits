#!/usr/bin/env python3
"""Execute all five notebooks with one interpreter and isolated run artifacts."""

import argparse
from datetime import datetime
import hashlib
from importlib import import_module, metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
STAGES = [
    ("data_processing", "pipeline_notebooks/data_processing.ipynb"),
    ("classification", "pipeline_notebooks/classification_model.ipynb"),
    ("regression", "pipeline_notebooks/regression_model.ipynb"),
    ("regression_high", "pipeline_notebooks/regression_70to100_model.ipynb"),
    ("combined", "pipeline_notebooks/combined_pipeline.ipynb"),
]
DEPENDENCIES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
    "xgboost": "xgboost", "category-encoders": "category_encoders", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "joblib": "joblib", "nbformat": "nbformat", "nbclient": "nbclient",
    "ipykernel": "ipykernel", "threadpoolctl": "threadpoolctl",
}


def now():
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    for key in ["cv_folds", "n_jobs", "native_threads", "classification_iterations",
                "regression_iterations", "high_regression_iterations"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["cv_folds"] < 2 or not 0 < config["test_size"] < 1:
        raise ValueError("Require at least two folds and 0 < test_size < 1")
    if config["sample_rows"] is not None and config["sample_rows"] < 100:
        raise ValueError("sample_rows must be null or at least 100")
    if not config["classifier_models"] or not config["regressor_models"]:
        raise ValueError("Require classifier and regressor model lists")


def dependency_check():
    result = {"python": sys.version, "executable": sys.executable,
              "virtual_environment": sys.prefix != sys.base_prefix,
              "platform": platform.platform(), "packages": {}, "errors": {}}
    for distribution, module in DEPENDENCIES.items():
        try:
            import_module(module)
            result["packages"][distribution] = metadata.version(distribution)
        except Exception as error:
            result["errors"][distribution] = f"{type(error).__name__}: {error}"
    if sys.platform == "darwin":
        for runtime in [Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
                        Path("/usr/local/opt/libomp/lib/libomp.dylib")]:
            if runtime.exists():
                result["openmp_runtime"] = {"path": str(runtime.resolve()), "sha256": file_hash(runtime)}
                break
    return result


def estimate_search(config):
    from sklearn.model_selection import ParameterGrid
    from pipeline_models import model_specs
    result = []
    for stage, classification, budget_key in [
        ("classification", True, "classification_iterations"),
        ("regression", False, "regression_iterations"),
        ("regression_high", False, "high_regression_iterations"),
    ]:
        for name, (_, grid) in model_specs(classification, config).items():
            candidates = min(config[budget_key], len(ParameterGrid(grid)))
            result.append({"stage": stage, "model": name, "candidates": candidates,
                           "fits_including_refit": candidates * config["cv_folds"] + 1})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full", "original_budget"], default="smoke")
    parser.add_argument("--config", type=Path, help="Use a custom JSON configuration")
    parser.add_argument("--n-jobs", type=int, help="Parallel search workers; default 2")
    parser.add_argument("--check", action="store_true", help="Verify dependencies and print the plan without executing notebooks")
    parser.add_argument("--run-dir", type=Path, help="New output directory; existing directories are refused")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / f"{args.profile}.json"
    config = json.loads(config_path.read_text())
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs
    validate_config(config)
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (args.run_dir or ROOT / "runs" / f"{stamp}_{config['profile']}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ["logs", "executed_notebooks", "models", "metrics", "plots", "data",
                   "predictions", "cv_results", "source", "runtime"]:
        (run_dir / folder).mkdir()
    env = os.environ.copy()
    env.update({
        "SPOTIFY_RUN_DIR": str(run_dir), "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(run_dir / "runtime" / "matplotlib"),
        "IPYTHONDIR": str(run_dir / "runtime" / "ipython"),
        "JUPYTER_CONFIG_DIR": str(run_dir / "runtime" / "jupyter_config"),
        "JUPYTER_RUNTIME_DIR": str(run_dir / "runtime" / "jupyter"),
        "JOBLIB_TEMP_FOLDER": str(run_dir / "runtime" / "joblib"),
        "XDG_CACHE_HOME": str(run_dir / "runtime" / "cache"),
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONHASHSEED": str(config["seed"]), "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": str(config["native_threads"]),
        "OMP_NUM_THREADS": str(config["native_threads"]),
        "MKL_NUM_THREADS": str(config["native_threads"]),
        "VECLIB_MAXIMUM_THREADS": str(config["native_threads"]),
    })
    os.environ.update(env)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "pipeline.log"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("runner")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    write_json(run_dir / "config.json", config)
    manifest = {"status": "checking", "started_at": now(), "profile": config["profile"],
                "run_dir": str(run_dir), "config_source": str(config_path.resolve()),
                "config": config, "stages": [], "source_sha256": {}}
    write_json(run_dir / "run_manifest.json", manifest)
    started = time.monotonic()
    log.info("Run directory: %s", run_dir)
    try:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Use the project virtual environment: .venv/bin/python run_pipeline.py ...")
        checked = dependency_check()
        write_json(run_dir / "environment.json", checked)
        if checked["errors"]:
            raise RuntimeError(f"Dependency import failures: {checked['errors']}")
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (run_dir / "requirements.freeze.txt").write_text(frozen)
        log.info("Verified %d dependencies in %s", len(checked["packages"]), sys.executable)
        source_files = [ROOT / name for _, name in STAGES]
        source_files += [ROOT / "pipeline_lib.py", ROOT / "pipeline_models.py", ROOT / "run_pipeline.py",
                         ROOT / "requirements.txt", ROOT / "requirements.lock.txt"]
        for path in source_files:
            if path.exists():
                (run_dir / "source" / path.name).write_bytes(path.read_bytes())
                manifest["source_sha256"][str(path.relative_to(ROOT))] = file_hash(path)
        manifest["input_sha256"] = {
            name: file_hash(ROOT / name)
            for name in ["spotify_tracks_dataset.csv", "spotify_tracks_dataset_processed.csv"]
        }
        search_plan = estimate_search(config)
        write_json(run_dir / "search_plan.json", search_plan)
        log.info("Search budget: %d model fits, %d folds, %d workers; sample_rows=%s",
                 sum(x["fits_including_refit"] for x in search_plan),
                 config["cv_folds"], config["n_jobs"], config["sample_rows"] or "all")
        if args.check:
            manifest["status"] = "checked_only"
            log.info("Checks passed. No notebooks or model training executed.")
            return 0
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager

        kernel_root = run_dir / "runtime" / "kernels"
        kernel_dir = kernel_root / "spotify-pipeline"
        kernel_dir.mkdir(parents=True)
        write_json(kernel_dir / "kernel.json", {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Spotify project .venv", "language": "python",
        })
        manifest["status"] = "running"
        for stage_name, notebook_name in STAGES:
            stage = {"stage": stage_name, "notebook": notebook_name, "status": "running", "started_at": now()}
            manifest["stages"].append(stage)
            write_json(run_dir / "run_manifest.json", manifest)
            stage_start = time.monotonic()
            notebook = nbformat.read(run_dir / "source" / Path(notebook_name).name, as_version=4)
            destination = run_dir / "executed_notebooks" / Path(notebook_name).name
            output_path = run_dir / "logs" / f"{stage_name}_notebook.log"
            manager = KernelManager(
                kernel_name="spotify-pipeline",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
            )

            def cell_finished(cell, cell_index, **kwargs):
                with output_path.open("a") as handle:
                    handle.write(f"\nCELL {cell_index}\n")
                    for output in cell.get("outputs", []):
                        if output.output_type == "stream":
                            handle.write(output.text)
                        elif output.output_type == "error":
                            handle.write("\n".join(output.traceback))
                        elif "text/plain" in output.get("data", {}):
                            handle.write(output.data["text/plain"] + "\n")
                nbformat.write(notebook, destination)
                log.info("%s: cell %d finished", stage_name, cell_index)

            client = NotebookClient(
                notebook, km=manager, timeout=config["cell_timeout_seconds"],
                allow_errors=False, record_timing=True, on_cell_executed=cell_finished,
            )
            log.info("START NOTEBOOK %s", notebook_name)
            try:
                with (run_dir / "logs" / f"{stage_name}_kernel.log").open("w") as kernel_output:
                    client.execute(cwd=str(ROOT), env=env, stdout=kernel_output,
                                   stderr=subprocess.STDOUT, cleanup_kc=True)
                stage["status"] = "completed"
            except BaseException:
                stage["status"] = "failed"
                raise
            finally:
                nbformat.write(notebook, destination)
                stage["finished_at"] = now()
                stage["seconds"] = time.monotonic() - stage_start
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                write_json(run_dir / "run_manifest.json", manifest)
            log.info("DONE NOTEBOOK %s (%.1fs)", notebook_name, stage["seconds"])
        for name, previous in manifest["source_sha256"].items():
            if file_hash(ROOT / name) != previous:
                raise RuntimeError(f"Source changed during execution: {name}")
        for name, previous in manifest["input_sha256"].items():
            if file_hash(ROOT / name) != previous:
                raise RuntimeError(f"Original input changed during execution: {name}")
        manifest["status"] = "completed"
        log.info("All notebooks completed. Results: %s", run_dir / "RESULTS.md")
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        log.error("Interrupted; completed artifacts remain in %s", run_dir)
        return 130
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        (run_dir / "logs" / "failure.log").write_text(traceback.format_exc())
        log.exception("Pipeline stopped: %s", error)
        return 1
    finally:
        manifest["finished_at"] = now()
        manifest["seconds"] = time.monotonic() - started
        write_json(run_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    sys.exit(main())
