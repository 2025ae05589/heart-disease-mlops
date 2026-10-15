"""Functions for the MLflow model registry.

A tested model is registered as a new version with the alias challenger. If
it is better than the current champion, the alias champion is moved to it.
The champion is always found by its alias, never by a version number.

Used by step 5 (register) and step 6 (export) of the pipeline.
"""
import subprocess

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import CLEAN_PATH, METRICS, MODEL_NAME, ROOT
from src.tracking.experiments import file_hash, setup_mlflow


def git_commit():
    """Return the short hash of the current git commit, or "unknown" if git
    is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def register_challenger(run_id, test_metrics):
    """Register the trained model of a run as a new version and give it the
    alias challenger.

    run_id: parent run of the selected model.
    test_metrics: its metrics on the test set.

    The test metrics, the hash of the cleaned data file and the git commit
    are stored as tags on the version. Returns the version number.
    """
    setup_mlflow()
    client = MlflowClient()

    # MLflow keeps the trained model of a run under its own model id
    model_id = client.get_run(run_id).outputs.model_outputs[0].model_id
    model_version = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)
    number = model_version.version
    client.set_registered_model_alias(MODEL_NAME, "challenger", number)

    for metric in METRICS:
        client.set_model_version_tag(
            MODEL_NAME, number, f"test_{metric}", f"{test_metrics[metric]:.4f}"
        )
    client.set_model_version_tag(MODEL_NAME, number, "data_sha256", file_hash(CLEAN_PATH))
    client.set_model_version_tag(MODEL_NAME, number, "git_commit", git_commit())
    return number


def get_champion():
    """Return the model version that has the alias champion, or None if no
    champion has been set yet."""
    setup_mlflow()
    try:
        return MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None


def champion_test_roc_auc():
    """Return the test-set ROC-AUC of the champion, read from its MLflow run.
    None if there is no champion. Used for gate 2."""
    champion = get_champion()
    if champion is None:
        return None
    run = MlflowClient().get_run(champion.run_id)
    return run.data.metrics["test_roc_auc"]


def promote_to_champion(number):
    """Make this version the champion: the alias champion is moved to it and
    its alias challenger is removed."""
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", number)
    client.delete_registered_model_alias(MODEL_NAME, "challenger")


def mark_gate_failed(number, gate):
    """Tag this version with the name of the gate it failed, for example
    "gate 2". The version stays in the registry as challenger."""
    MlflowClient().set_model_version_tag(MODEL_NAME, number, "failed_gate", gate)
