"""Loading of the champion model and prediction for new patients.

The model and its details are read from models/champion/, the folder to which
the pipeline exports the champion from the MLflow registry.

This module contains functions which are called by api/main.py and notebooks/03_inference.ipynb.
"""
from dataclasses import dataclass
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import yaml

from src.data.data_dictionary import FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "champion"

LABELS = {0: "no disease", 1: "disease"}


@dataclass
class LoadedModel:
    """The loaded champion model and its registry metadata, returned by
    load_model()."""
    model: object       # trained scikit-learn pipeline
    name: str           # registered model name
    version: str        # registered model version
    run_id: str         # MLflow run that trained the model
    trained_at: str     # training time (UTC)


def load_model(model_dir: Path = MODEL_DIR) -> LoadedModel:
    """Load the exported model from its folder and read its details.

    The name and version are in the file registered_model_meta, written by
    MLflow when the model is exported from the registry. The run ID and the
    training time are in the file MLmodel.
    """
    model_dir = Path(model_dir)
    registry = yaml.safe_load((model_dir / "registered_model_meta").read_text())
    details = yaml.safe_load((model_dir / "MLmodel").read_text())

    return LoadedModel(
        model=mlflow.sklearn.load_model(str(model_dir)),
        name=registry["model_name"],
        version=str(registry["model_version"]),
        run_id=details["run_id"],
        # fractional seconds are omitted
        trained_at=str(details["utc_time_created"]).split(".")[0] + " UTC",
    )


def predict(model, patients: list[dict]) -> list[dict]:
    """Return the prediction, label, confidence and imputed features for each
    patient. None values are imputed by the pipeline; the threshold is 0.5."""
    # columns in the order used in training; None becomes a missing value
    data = pd.DataFrame(patients, columns=FEATURES).astype(float)
    probabilities = model.predict_proba(data)[:, 1]

    results = []
    for patient, probability in zip(patients, probabilities):
        prediction = int(probability >= 0.5)
        confidence = probability if prediction == 1 else 1 - probability

        filled = []
        for name in FEATURES:
            if patient[name] is None:
                filled.append(name)

        results.append({
            "prediction": prediction,
            "label": LABELS[prediction],
            "confidence": round(float(confidence), 4),
            "filled_fields": filled,
        })
    return results
