"""Preprocessing steps of the model pipeline.

These are the steps that learn something from the data: the fill values for
missing cells, the mean and spread used for scaling, and the list of
categories used for encoding. They are kept inside the model pipeline so that
they are learnt from the training rows only and are applied in the same way
during training, testing and in the API.

The choices made here come from the EDA notebook
(notebooks/01_data_acquisition_and_eda.ipynb).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_prep import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """Return the preprocessing steps (not yet fitted).

    Numerical features:
      1. A missing value is filled with the median of the column.
      2. The values are standardised (mean 0, standard deviation 1).

    Categorical features:
      1. A missing value is filled with the most common value of the column.
      2. Each category becomes its own 0/1 column (one-hot encoding).
    """
    numeric_steps = Pipeline([
        # The data has no missing numerical value today, but a request to
        # the API may have one. The median is not pulled by extreme values.
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        # An average has no meaning for a code, so the most common value
        # is used.
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a category that was not seen in training
        # gets 0 in all its columns, so the pipeline does not fail on it.
        # This matters for rare categories such as restecg = 1 (4 patients).
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_steps, NUMERIC),
            ("categorical", categorical_steps, CATEGORICAL),
        ],
        # Keeps the output column names short, e.g. "age" and "cp_4.0".
        verbose_feature_names_out=False,
    )
