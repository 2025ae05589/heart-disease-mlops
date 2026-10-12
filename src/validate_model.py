"""Check that the exported champion model can be loaded and used on its own,
without the MLflow store.

Run from the project root, after the training:
    python -m src.validate_model
"""
import logging
import sys
from pathlib import Path

import mlflow.sklearn
import numpy as np
import yaml
from mlflow.models import Model

from src.tracking import EXPORT_DIR

log = logging.getLogger("validate_model")


def validate(model_dir: Path = EXPORT_DIR):
    """Load the model from its folder and predict for the sample patients
    saved with it. Returns the model name, its version and the probabilities.
    An error is raised if a check fails."""
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # another folder can be given, e.g. the model folder inside a Docker image
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPORT_DIR

    try:
        name, version, probability = validate(model_dir)
    except Exception as error:
        # any failure must stop the pipeline
        log.error("Model check failed for %s: %r", model_dir, error)
        sys.exit(1)

    log.info("Loaded %s version %s from %s", name, version, model_dir)
    log.info(
        "Model check passed: %d sample patients, probabilities %s",
        len(probability), np.round(probability, 3).tolist(),
    )


if __name__ == "__main__":
    main()
