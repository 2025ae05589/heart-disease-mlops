"""Common functions for recording experiments in MLflow."""
import hashlib
import os
import shutil
import subprocess
import warnings
from importlib.metadata import version

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.base import clone

from src.data_prep import CLEAN_PATH, SOURCE_PATH
from src.evaluation import (
    CV_FOLDS,
    METRICS,
    RANDOM_STATE,
    ROC_AUC_GATE,
    ROOT,
    TUNING_METRIC,
    cross_val_probabilities,
    passes_quality_gate,
    plot_confusion_matrix,
    plot_roc_curve,
)

# A local file by default. A shared MLflow server is used instead when its
# address is given in the environment variable MLFLOW_TRACKING_URI.
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix()
)
# where the artifacts of the local store are kept
ARTIFACT_DIR = ROOT / "mlruns"
EXPERIMENT = "Heart Disease Prediction"

# name in the model registry; the aliases "challenger" and "champion" point
# to versions of it
MODEL_NAME = "heart-disease-classifier"
# the champion is copied here for the API and the Docker image
EXPORT_DIR = ROOT / "models" / "champion"

# skops, the format MLflow saves in, refuses types it does not know.
TRUSTED_TYPES = [
    "numpy.dtype",                    # preprocessing steps
    "sklearn.tree._tree.Tree",        # Random Forest, Gradient Boosting
    "xgboost.core.Booster",           # XGBoost
    "xgboost.sklearn.XGBClassifier",  # XGBoost
]

# Listed here, otherwise MLflow takes about 14 seconds per model to find them.
MODEL_REQUIREMENTS = []
for package in ["scikit-learn", "xgboost", "pandas", "numpy", "skops"]:
    MODEL_REQUIREMENTS.append(f"{package}=={version(package)}")

# The warning is about the prediction column, which is never missing.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        if MLFLOW_URI.startswith("sqlite:"):
            mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
        else:
            # a shared server keeps the artifacts in its own storage
            mlflow.create_experiment(EXPERIMENT)
    mlflow.set_experiment(EXPERIMENT)


def file_hash(path):
    """SHA-256 hash of a data file. Line endings are made the same first,
    as they differ between Windows and Linux."""
    content = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def log_metrics(row):
    """Log the mean and std of every metric from one row of the metrics table."""
    for metric in METRICS:
        mlflow.log_metric(f"cv_{metric}_mean", row[f"{metric}_mean"])
        mlflow.log_metric(f"cv_{metric}_std", row[f"{metric}_std"])


def log_plots(pipeline, X_train, y_train, folds, title):
    """Log the ROC curve and the confusion matrix in the active run."""
    probability = cross_val_probabilities(pipeline, X_train, y_train, folds)

    fig = plot_roc_curve(y_train, probability, f"ROC curve - {title}")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)

    fig = plot_confusion_matrix(y_train, probability, f"Confusion matrix - {title}")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)


def log_trained_model(pipeline, X_train, y_train):
    """Train the pipeline on all the training patients and log it."""
    pipeline.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=X_train.head(5),
        pip_requirements=MODEL_REQUIREMENTS,
        skops_trusted_types=TRUSTED_TYPES,
    )


def log_combination_run(model_name, pipeline, settings, row, is_best,
                        X_train, y_train, folds, tags=None):
    """Log one combination of settings as a child run."""
    parts = []
    for name, value in settings.items():
        parts.append(f"{name}={value}")
    title = f"{model_name} ({', '.join(parts)})"

    with mlflow.start_run(run_name=title, nested=True):
        mlflow.set_tag("run_type", "combination")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("best_of_model", str(is_best))
        mlflow.set_tags(tags or {})

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(settings)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_STATE)

        log_metrics(row)
        log_plots(pipeline, X_train, y_train, folds, title)


def log_model_summary(model_name, pipeline, metrics_table, best_index,
                      X_train, y_train, folds):
    """Log the summary of a model in its parent run."""
    best_row = metrics_table.loc[best_index]

    mlflow.set_tag("run_type", "model")
    mlflow.set_tag("model_name", model_name)

    mlflow.log_param("model_type", model_name)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("tuning_metric", TUNING_METRIC)
    mlflow.log_param("combinations_tried", len(metrics_table))
    mlflow.log_param("source_file", SOURCE_PATH.as_posix())
    mlflow.log_param("source_sha256", file_hash(SOURCE_PATH))
    mlflow.log_param("data_file", CLEAN_PATH.as_posix())
    mlflow.log_param("data_sha256", file_hash(CLEAN_PATH))
    mlflow.log_param("training_patients", len(X_train))

    for column in metrics_table.columns:
        if not column.endswith("_mean") and not column.endswith("_std"):
            # read from the column, so that a whole number stays whole
            mlflow.log_param(f"best_{column}", metrics_table.at[best_index, column])

    log_metrics(best_row)
    mlflow.log_text(metrics_table.to_csv(index=False), "cv_results.csv")
    log_plots(pipeline, X_train, y_train, folds, f"{model_name} (best setting)")
    log_trained_model(pipeline, X_train, y_train)


def log_model_runs(model_name, pipeline, search, metrics_table,
                   X_train, y_train, folds, tags=None):
    """Log one model: a parent run and a child run for each combination.

    search is from run_grid_search() and metrics_table is from
    build_metrics_table(). tags are extra labels set on every run.
    """
    setup_mlflow()
    all_params = search.cv_results_["params"]
    best_index = search.best_index_

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags(tags or {})

        for i, row in metrics_table.iterrows():
            settings = {}
            for name, value in all_params[i].items():
                settings[name.replace("model__", "")] = value

            candidate = clone(pipeline).set_params(**all_params[i])
            log_combination_run(
                model_name, candidate, settings, row, i == best_index,
                X_train, y_train, folds, tags,
            )

        best_pipeline = clone(pipeline).set_params(**search.best_params_)
        log_model_summary(
            model_name, best_pipeline, metrics_table, best_index,
            X_train, y_train, folds,
        )

    print(f"Logged to MLflow: {model_name}, {len(metrics_table)} combinations.")


def load_model_results() -> pd.DataFrame:
    """One row per model, read from the parent runs in MLflow: the best
    setting, its cross-validation metrics and the run ID."""
    setup_mlflow()
    runs = mlflow.search_runs(
        filter_string="tags.run_type = 'model'",
        order_by=["attributes.start_time ASC"],
        output_format="list",
    )

    rows = {}
    for run in runs:
        # runs of side experiments are not models to compare
        if "experiment" in run.data.tags:
            continue

        settings = []
        for name, value in run.data.params.items():
            if name.startswith("best_"):
                settings.append(f"{name.replace('best_', '')}={value}")

        row = {
            "model": run.data.tags["model_name"],
            "best_setting": ", ".join(sorted(settings)),
            "run_id": run.info.run_id,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = run.data.metrics[f"cv_{metric}_mean"]
            row[f"{metric}_std"] = run.data.metrics[f"cv_{metric}_std"]

        # if a model was run more than once, its latest run replaces the earlier one
        rows[row["model"]] = row

    return pd.DataFrame(list(rows.values()))


def load_trained_model(run_id):
    """The trained model saved in a parent run."""
    setup_mlflow()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def log_test_results(run_id, model_name, test_metrics, y_test, probability):
    """Add the test-set metrics and plots to the parent run of the selected
    model."""
    setup_mlflow()
    title = f"{model_name} (test set)"

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("selected_model", "true")
        mlflow.set_tag("quality_gate_passed", str(passes_quality_gate(test_metrics)))

        mlflow.log_param("test_patients", len(y_test))
        mlflow.log_param("roc_auc_gate", ROC_AUC_GATE)

        for metric in METRICS:
            mlflow.log_metric(f"test_{metric}", test_metrics[metric])

        fig = plot_roc_curve(y_test, probability, f"ROC curve - {title}")
        mlflow.log_figure(fig, "plots/test_roc_curve.png")
        plt.close(fig)

        fig = plot_confusion_matrix(y_test, probability, f"Confusion matrix - {title}")
        mlflow.log_figure(fig, "plots/test_confusion_matrix.png")
        plt.close(fig)

    print(f"Logged to MLflow: test results of {model_name}.")


def git_commit():
    """Short hash of the current git commit, or "unknown" if git is not there."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def register_challenger(run_id, test_metrics):
    """Register the trained model of a parent run as a new version, with the
    alias challenger. Returns the version number."""
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
    """The model version with the alias champion, or None if there is none."""
    setup_mlflow()
    try:
        return MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None


def champion_test_roc_auc():
    """ROC-AUC of the champion on the test set, or None if there is no
    champion."""
    champion = get_champion()
    if champion is None:
        return None
    run = MlflowClient().get_run(champion.run_id)
    return run.data.metrics["test_roc_auc"]


def promote_to_champion(number):
    """Move the alias champion to this version."""
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", number)
    client.delete_registered_model_alias(MODEL_NAME, "challenger")


def mark_gate_failed(number, gate):
    """Record on the version which gate it failed."""
    MlflowClient().set_model_version_tag(MODEL_NAME, number, "failed_gate", gate)


def export_champion():
    """Copy the champion out of MLflow into EXPORT_DIR. Returns its version
    number."""
    setup_mlflow()
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@champion", dst_path=str(EXPORT_DIR)
    )
    return get_champion().version
