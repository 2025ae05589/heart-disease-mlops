"""Common functions for recording experiments in MLflow."""
import hashlib
import warnings
from importlib.metadata import version

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.base import clone

from src.data_prep import CLEAN_PATH, SOURCE_PATH
from src.evaluation import (
    CV_FOLDS,
    METRICS,
    RANDOM_STATE,
    ROOT,
    TUNING_METRIC,
    cross_val_probabilities,
    plot_confusion_matrix,
    plot_roc_curve,
)

MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
ARTIFACT_DIR = ROOT / "mlruns"
EXPERIMENT = "Heart Disease Prediction"

# skops, the format MLflow saves in, refuses types it does not know.
# This one is used by the preprocessing steps.
TRUSTED_TYPES = ["numpy.dtype"]

# Listed here, otherwise MLflow takes about 14 seconds per model to find them.
MODEL_REQUIREMENTS = []
for package in ["scikit-learn", "pandas", "numpy", "skops"]:
    MODEL_REQUIREMENTS.append(f"{package}=={version(package)}")

# The warning is about the prediction column, which is never missing.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
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
            mlflow.log_param(f"best_{column}", best_row[column])

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
