"""Packaging of the champion model for deployment.

The champion is copied out of the MLflow registry into the folder
models/champion/. The API and the Docker image will load the model from
this folder. So they do not need the MLflow database (mlflow.db and mlruns/).

The folder is also checked here: the model is loaded from it and predicts
for the sample patients saved with it.

Used by steps 6 and 7 of the pipeline.
"""
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from mlflow.models import Model

from src.config import EXPORT_DIR, MODEL_NAME
from src.tracking.registry import get_champion


def export_champion(export_dir: Path = EXPORT_DIR):
    """Copy the champion out of the MLflow registry into export_dir. What
    was in the folder before is removed. Returns the version number."""
    number = get_champion().version
    if export_dir.exists():
        shutil.rmtree(export_dir)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@champion", dst_path=str(export_dir)
    )
    return number


def validate_exported_model(model_dir: Path = EXPORT_DIR):
    """Load the model from its folder, without the MLflow database, and
    predict for the sample patients saved with it.

    Returns the model name, its version and the predicted probabilities. An
    error is raised if the model cannot be loaded or a probability is not
    valid.
    """
    # written by MLflow when the model is exported from the registry
    details = yaml.safe_load((model_dir / "registered_model_meta").read_text())
    name, version = details["model_name"], details["model_version"]

    model = mlflow.sklearn.load_model(str(model_dir))
    sample = Model.load(str(model_dir)).load_input_example(str(model_dir))

    probability = model.predict_proba(sample)[:, 1]
    if len(probability) != len(sample):
        raise ValueError(f"{len(probability)} predictions for {len(sample)} patients")
    if not np.isfinite(probability).all():
        raise ValueError("a predicted probability is not a number")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("a predicted probability is outside 0 to 1")

    return name, version, probability
