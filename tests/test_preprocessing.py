"""Tests for src/features/preprocessing.py: the preprocessing steps.

It is checked that missing values are filled, numerical features are
standardised, categorical features are one-hot encoded, and a category not
seen in training does not cause an error. Only the training patients are
used.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import CATEGORICAL, NUMERIC
from src.data.preparation import split_data
from src.features.preprocessing import build_preprocessor, make_pipeline


@pytest.fixture(scope="module")
def X_train():
    """Features of the training patients."""
    return split_data()[0]


@pytest.fixture(scope="module")
def preprocessor(X_train):
    """The preprocessing steps fitted on the training patients."""
    return build_preprocessor().set_output(transform="pandas").fit(X_train)


def test_output_has_no_missing_values(preprocessor, X_train):
    rows = X_train.head(4).copy()
    rows.iloc[0, rows.columns.get_loc("chol")] = np.nan
    rows.iloc[1, rows.columns.get_loc("thal")] = np.nan

    output = preprocessor.transform(rows)

    assert not output.isna().any().any()


def test_numerical_features_are_standardised(preprocessor, X_train):
    output = preprocessor.transform(X_train)

    assert np.allclose(output[NUMERIC].mean(), 0, atol=1e-9)
    assert np.allclose(output[NUMERIC].std(ddof=0), 1, atol=1e-9)


def test_categorical_features_are_one_hot_encoded(preprocessor, X_train):
    output = preprocessor.transform(X_train)
    encoded = output.drop(columns=NUMERIC)

    # one column for each value of each categorical feature
    values = sum(X_train[feature].nunique() for feature in CATEGORICAL)
    assert encoded.shape[1] == values
    assert set(np.unique(encoded)) == {0.0, 1.0}
    # each patient has one value per categorical feature
    assert (encoded.sum(axis=1) == len(CATEGORICAL)).all()


def test_unseen_category_does_not_fail(preprocessor, X_train):
    row = X_train.head(1).copy()
    row["cp"] = 9.0

    output = preprocessor.transform(row)

    cp_columns = [column for column in output.columns if column.startswith("cp_")]
    assert output[cp_columns].sum(axis=1).iloc[0] == 0


def test_pipeline_steps():
    pipeline = make_pipeline(LogisticRegression())

    assert list(pipeline.named_steps) == ["preprocess", "model"]
