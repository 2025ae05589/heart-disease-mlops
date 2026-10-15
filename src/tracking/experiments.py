"""Functions for recording experiments in MLflow and reading them back.

Each model gets one parent run, and one child run under it for every
combination of its settings. The parent run holds the best setting, its
metrics and plots, the trained model and the hashes of the data files. The
test-set results are added later to the parent run of the selected model.

Used by steps 3, 4 and 5 of the pipeline and by the notebook.
"""
import hashlib
import warnings
from importlib.metadata import version

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone

from src.config import (
    ARTIFACT_DIR,
    CLEAN_PATH,
    CV_FOLDS,
    EXPERIMENT,
    METRICS,
    MLFLOW_URI,
    RANDOM_STATE,
    ROC_AUC_GATE,
    ROOT,
    SOURCE_PATH,
    TUNING_METRIC,
)
from src.models.evaluation import (
    cross_val_probabilities,
    passes_quality_gate,
    plot_confusion_matrix,
    plot_roc_curve,
)

# skops, the format MLflow saves in, refuses types it does not know.
TRUSTED_TYPES = [
    "numpy.dtype",                    # preprocessing steps
    "sklearn.tree._tree.Tree",        # Random Forest, Gradient Boosting
    "xgboost.core.Booster",           # XGBoost
    "xgboost.sklearn.XGBClassifier",  # XGBoost
]

# Packages needed to load any of the models. They are listed here, otherwise
# MLflow takes about 14 seconds per model to find them.
BASE_PACKAGES = ["scikit-learn", "pandas", "numpy", "skops"]

# The warning is about the prediction column, which is never missing.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def setup_mlflow():
    """Point MLflow to the store given in MLFLOW_URI and select the
    experiment. The experiment is created on first use."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        if MLFLOW_URI.startswith("sqlite:"):
            mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
        else:
            # a shared server keeps the artifacts in its own storage
            mlflow.create_experiment(EXPERIMENT)
    mlflow.set_experiment(EXPERIMENT)


def file_hash(path):
    """Return the SHA-256 hash of a data file. Line endings are made the same
    first, as they differ between Windows and Linux."""
    content = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def log_metrics(row):
    """Log the cross-validation mean and std of every metric in the active
    run. row: one row of the metrics table."""
    for metric in METRICS:
        mlflow.log_metric(f"cv_{metric}_mean", row[f"{metric}_mean"])
        mlflow.log_metric(f"cv_{metric}_std", row[f"{metric}_std"])


def log_plots(pipeline, X_train, y_train, folds, title):
    """Log the ROC curve and the confusion matrix in the active run. Both
    are made from cross-validated predictions on the training patients."""
    probability = cross_val_probabilities(pipeline, X_train, y_train, folds)

    fig = plot_roc_curve(y_train, probability, f"ROC curve - {title}")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)

    fig = plot_confusion_matrix(y_train, probability, f"Confusion matrix - {title}")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)


def model_requirements(pipeline):
    """Return the packages needed to load this model, with the versions used
    in training. xgboost is listed only for a model that uses it, so that a
    Docker image of another model does not have to install it."""
    packages = list(BASE_PACKAGES)
    if type(pipeline.named_steps["model"]).__module__.startswith("xgboost"):
        packages.append("xgboost")

    requirements = []
    for package in packages:
        requirements.append(f"{package}=={version(package)}")
    return requirements


def log_trained_model(pipeline, X_train, y_train):
    """Train the pipeline on all the training patients and log it in the
    active run, with five patients as the sample input."""
    pipeline.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=X_train.head(5),
        pip_requirements=model_requirements(pipeline),
        skops_trusted_types=TRUSTED_TYPES,
    )


def log_combination_run(model_name, pipeline, settings, row, is_best,
                        X_train, y_train, folds, tags=None):
    """Log one combination of settings as a child run: its settings, metrics
    and plots.

    row: the row of this combination in the metrics table.
    is_best: True for the best combination of the model.
    """
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
    """Log the summary of a model in its parent run: the best setting, its
    metrics and plots, the table of all combinations, the hashes of the data
    files and the trained model.

    pipeline: the pipeline with the best setting, not yet fitted.
    best_index: row of the best combination in the metrics table.
    """
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
    """Return a table with one row per model, read from the parent runs in
    MLflow: the best setting, its cross-validation metrics and the run ID.
    If a model was run more than once, its latest run is taken."""
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
    """Return the trained model saved in a parent run."""
    setup_mlflow()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def log_test_results(run_id, model_name, test_metrics, y_test, probability):
    """Add the test-set results to the parent run of the selected model: the
    metrics, the plots, the tag selected_model and whether gate 1 passed.

    run_id: parent run of the selected model.
    probability: predicted probability of disease for the test patients.
    """
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


def load_selected_model():
    """Return the name, run ID and test-set metrics of the selected model.
    It is found by the tag selected_model among the latest runs. None if no
    model has been tested yet."""
    # the client takes the MLflow address that is set at the time it is made
    setup_mlflow()
    client = MlflowClient()
    for _, row in load_model_results().iterrows():
        run = client.get_run(row["run_id"])
        if run.data.tags.get("selected_model") == "true":
            test_metrics = {}
            for metric in METRICS:
                test_metrics[metric] = run.data.metrics[f"test_{metric}"]
            return row["model"], row["run_id"], test_metrics
    return None
