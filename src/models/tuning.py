"""Tuning of one model with cross-validation.

A grid search tries every combination of settings of the model on the same
stratified folds. The results are put in a table with one row for each
combination.

Used by step 3 of the pipeline, the notebook and the tests.
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import CV_FOLDS, METRICS, RANDOM_STATE, TUNING_METRIC


def create_cv_folds():
    """Return the stratified cross-validation folds. The seed is fixed, so
    every model is compared on the same folds."""
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


def run_grid_search(pipeline, grid, X_train, y_train, folds):
    """Try every combination in the grid with cross-validation on the
    training patients.

    All the metrics are computed for each combination. The best combination
    is the one with the highest mean TUNING_METRIC. Returns the fitted
    search.
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
    """Return a table with one row per combination: its settings, and the
    mean and std of each metric over the folds.

    search: the fitted search from run_grid_search().
    """
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
    """Return the metrics table in a form for display: each metric as one
    "mean ± std" column, with three decimals."""
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
