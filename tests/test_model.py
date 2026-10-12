"""Tests for training and prediction with the model pipeline. Only the
training patients are used, so the test set stays unused."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    METRICS,
    RANDOM_STATE,
    build_metrics_table,
    create_cv_folds,
    cross_val_probabilities,
    run_grid_search,
    split_data,
)
from src.preprocessing import make_pipeline


@pytest.fixture(scope="module")
def training_data():
    X_train, _, y_train, _ = split_data()
    return X_train, y_train


@pytest.fixture(scope="module")
def pipeline():
    return make_pipeline(LogisticRegression(random_state=RANDOM_STATE))


@pytest.fixture(scope="module")
def trained(pipeline, training_data):
    return clone(pipeline).fit(*training_data)


def test_probabilities_are_valid(trained, training_data):
    X_train, _ = training_data

    probability = trained.predict_proba(X_train)

    assert probability.shape == (len(X_train), 2)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert np.allclose(probability.sum(axis=1), 1)


def test_prediction_is_no_disease_or_disease(trained, training_data):
    X_train, _ = training_data

    assert set(trained.predict(X_train)) == {0, 1}


def test_patient_with_missing_values_gets_a_prediction(trained, training_data):
    X_train, _ = training_data
    patient = X_train.head(1).copy()
    patient[["chol", "ca", "thal"]] = np.nan

    probability = trained.predict_proba(patient)[0, 1]

    assert 0 <= probability <= 1


def test_training_twice_gives_the_same_model(pipeline, trained, training_data):
    X_train, _ = training_data
    again = clone(pipeline).fit(*training_data)

    assert np.array_equal(
        again.predict_proba(X_train), trained.predict_proba(X_train)
    )


def test_cross_validated_probability_for_each_training_patient(pipeline, training_data):
    X_train, y_train = training_data

    probability = cross_val_probabilities(pipeline, X_train, y_train, create_cv_folds())

    assert probability.shape == (len(X_train),)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_grid_search_and_metrics_table(pipeline, training_data):
    X_train, y_train = training_data
    grid = {"model__C": [0.1, 1]}

    search = run_grid_search(pipeline, grid, X_train, y_train, create_cv_folds())
    table = build_metrics_table(search)

    assert table["C"].tolist() == [0.1, 1]
    for metric in METRICS:
        assert table[f"{metric}_mean"].between(0, 1).all()
        assert (table[f"{metric}_std"] >= 0).all()
    # the best combination is the one with the highest mean ROC-AUC
    assert search.best_index_ == table["roc_auc_mean"].idxmax()
    # the model learns something: clearly better than random guessing
    assert table["roc_auc_mean"].max() > 0.8
