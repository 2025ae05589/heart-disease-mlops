"""Common functions for evaluating a model: data split, cross-validation
folds, grid search, metrics table and plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.data_prep import CATEGORICAL, CLEAN_PATH, NUMERIC, TARGET

# Paths are built from the project folder, so the code runs the same way
# from the project root and from the notebooks folder.
ROOT = Path(__file__).resolve().parent.parent

FEATURES = NUMERIC + CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# name used here -> name used by scikit-learn
METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# decides the best combination of settings of a model
TUNING_METRIC = "roc_auc"


def split_data():
    """Return X_train, X_test, y_train, y_test (stratified 80/20 split)."""
    data = pd.read_csv(ROOT / CLEAN_PATH)

    # float, because an int column cannot hold a missing value (NaN),
    # and new patient data may have one
    X = data[FEATURES].astype(float)
    y = data[TARGET]

    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def create_cv_folds():
    """The stratified folds, the same for every model."""
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


def run_grid_search(pipeline, grid, X_train, y_train, folds):
    """Try every combination in the grid with cross-validation.

    All metrics are computed. TUNING_METRIC picks the best combination.
    """
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=METRICS,
        refit=TUNING_METRIC,
        cv=folds,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def build_metrics_table(search) -> pd.DataFrame:
    """One row per combination, with the mean and std of each metric."""
    results = search.cv_results_

    rows = []
    for i, params in enumerate(results["params"]):
        row = {}
        for name, value in params.items():
            # None is kept as text, otherwise the table shows it as NaN
            row[name.replace("model__", "")] = "None" if value is None else value
        for metric in METRICS:
            row[f"{metric}_mean"] = results[f"mean_test_{metric}"][i]
            row[f"{metric}_std"] = results[f"std_test_{metric}"][i]
        rows.append(row)

    return pd.DataFrame(rows)


def readable_metrics(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Metrics table for display, with each metric as one "mean ± std" column."""
    setting_names = []
    for column in metrics_table.columns:
        if not column.endswith("_mean") and not column.endswith("_std"):
            setting_names.append(column)

    table = metrics_table[setting_names].copy()
    for metric in METRICS:
        mean = metrics_table[f"{metric}_mean"].map("{:.3f}".format)
        std = metrics_table[f"{metric}_std"].map("{:.3f}".format)
        table[metric] = mean + " ± " + std
    return table


def cross_val_probabilities(pipeline, X_train, y_train, folds):
    """Cross-validated probability of disease for each training patient."""
    probabilities = cross_val_predict(
        pipeline, X_train, y_train, cv=folds, method="predict_proba"
    )
    return probabilities[:, 1]


def plot_roc_curve(y_true, probability, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, probability, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            label="random guessing (AUC = 0.5)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, probability, title):
    """Confusion matrix at the 0.5 cut-off."""
    prediction = (probability >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, prediction, display_labels=["no disease", "disease"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig
