"""Tests for src/models/tuning.py: the cross-validation folds, the grid
search and the metrics table.

A small grid search on Logistic Regression is run once and its table is
checked. Only the training patients are used.
"""
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import CV_FOLDS, METRICS, RANDOM_STATE
from src.data.preparation import split_data
from src.features.preprocessing import make_pipeline
from src.models.tuning import build_metrics_table, create_cv_folds, run_grid_search


@pytest.fixture(scope="module")
def training_data():
    """Features and target of the training patients."""
    X_train, _, y_train, _ = split_data()
    return X_train, y_train


@pytest.fixture(scope="module")
def search(training_data):
    """A fitted grid search on Logistic Regression."""
    X_train, y_train = training_data
    pipeline = make_pipeline(LogisticRegression(random_state=RANDOM_STATE))
    grid = {"model__C": [0.1, 1]}
    return run_grid_search(pipeline, grid, X_train, y_train, create_cv_folds())


def test_each_training_patient_is_validated_once(training_data):
    X_train, y_train = training_data
    folds = create_cv_folds()

    validated = []
    for _, validation_rows in folds.split(X_train, y_train):
        validated.extend(validation_rows)

    assert folds.get_n_splits() == CV_FOLDS
    assert sorted(validated) == list(range(len(X_train)))


def test_grid_search_and_metrics_table(search):
    table = build_metrics_table(search)

    assert table["C"].tolist() == [0.1, 1]
    for metric in METRICS:
        assert table[f"{metric}_mean"].between(0, 1).all()
        assert (table[f"{metric}_std"] >= 0).all()
    # the best combination is the one with the highest mean ROC-AUC
    assert search.best_index_ == table["roc_auc_mean"].idxmax()
    # the model performs clearly better than random guessing (ROC-AUC 0.5)
    assert table["roc_auc_mean"].max() > 0.8
