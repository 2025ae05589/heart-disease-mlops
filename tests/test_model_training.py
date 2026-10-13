"""Tests for training and prediction with the model pipeline.

A Logistic Regression pipeline is trained once and checked: valid
probabilities, a prediction for a patient with missing values, and the same
model when trained twice. Only the training patients are used, so the test
set stays unused.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.data.preparation import split_data
from src.features.preprocessing import make_pipeline
from src.models.evaluation import cross_val_probabilities
from src.models.tuning import create_cv_folds


@pytest.fixture(scope="module")
def training_data():
    """Features and target of the training patients."""
    X_train, _, y_train, _ = split_data()
    return X_train, y_train


@pytest.fixture(scope="module")
def pipeline():
    """A Logistic Regression pipeline, not yet fitted."""
    return make_pipeline(LogisticRegression(random_state=RANDOM_STATE))


@pytest.fixture(scope="module")
def trained(pipeline, training_data):
    """The pipeline trained on the training patients."""
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
