"""Loading of the champion model and prediction for new patients.

The model is read from the exported folder models/champion/, so the MLflow
database is not needed. The details of the model (name, version, run ID and
training time) are read from the files that MLflow keeps in that folder.

This file knows nothing about web requests. It is used by api/main.py.
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
    """The trained model together with its details."""
    model: object
    name: str
    version: str
    run_id: str
    trained_at: str


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
        # the fraction of a second is left out
        trained_at=str(details["utc_time_created"]).split(".")[0] + " UTC",
    )


def predict(model, patients: list[dict]) -> list[dict]:
    """Return one result for each patient: the prediction, its label, the
    confidence and the features that were filled.

    model: the trained pipeline.
    patients: one dict per patient with the 13 features. A value of None is
    a missing value, which the pipeline fills.

    The confidence is the probability of the predicted class. A probability
    of disease of 0.5 or above is predicted as disease.
    """
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
