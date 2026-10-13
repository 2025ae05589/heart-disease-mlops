"""Step 7: check that the exported champion can be loaded and used on its
own, without the MLflow database (mlflow.db and mlruns/).

The model is loaded from models/champion/ and predicts for the sample
patients saved with it. The step ends with an error if anything fails.

Run from the project root, after step 6:
    python -m pipeline.step7_validate_model

Another model folder can be given:
    python -m pipeline.step7_validate_model <folder>
"""
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import EXPORT_DIR, LOG_DATE_FORMAT, LOG_FORMAT
from src.packaging.export import validate_exported_model

log = logging.getLogger("step7_validate_model")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    # another folder can be given, e.g. the model folder inside a Docker image
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPORT_DIR

    try:
        name, version, probability = validate_exported_model(model_dir)
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
