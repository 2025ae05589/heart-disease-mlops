"""Preprocessing steps of the model pipeline, as decided in the EDA notebook."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_prep import CATEGORICAL, NUMERIC


def build_preprocessor(numeric=NUMERIC, categorical=CATEGORICAL) -> ColumnTransformer:
    """Return the preprocessing steps (not yet fitted).

    Numerical features: median imputation, then standardisation.
    Categorical features: most frequent value imputation, then one-hot
    encoding.
    """
    numeric_steps = Pipeline([
        # no numerical value is missing today, but new patient data may have one
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric),
            ("categorical", categorical_steps, categorical),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model, numeric=NUMERIC, categorical=CATEGORICAL) -> Pipeline:
    """Preprocessing followed by the model, as one pipeline."""
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", model),
    ])
