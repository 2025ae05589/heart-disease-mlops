"""Data quality checks on the raw Cleveland file.

The number of patients and columns, duplicate rows, valid codes, limits of
the numerical values and the places of the missing values are checked. They
repeat the checks of the EDA notebook (section 5), so that the pipeline
fails if the data changes.
"""
import pandas as pd
import pytest

from src.config import RAW_PATH, ROOT

# values allowed in the UCI data dictionary
VALID_CODES = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
    "num": [0, 1, 2, 3, 4],
}

# broad limits that only catch entry errors; they are not clinical limits
LIMITS = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0, 7),
}


@pytest.fixture(scope="module")
def raw():
    """The raw Cleveland file, read once for all the tests."""
    return pd.read_csv(ROOT / RAW_PATH)


def test_number_of_patients_and_columns(raw):
    assert raw.shape == (303, 14)
    assert set(raw.columns) == set(VALID_CODES) | set(LIMITS)


def test_no_duplicate_rows(raw):
    assert raw.duplicated().sum() == 0


@pytest.mark.parametrize("column", list(VALID_CODES))
def test_codes_are_valid(raw, column):
    # missing values are checked separately
    values = raw[column].dropna()

    assert values.isin(VALID_CODES[column]).all()


@pytest.mark.parametrize("column", list(LIMITS))
def test_values_are_within_limits(raw, column):
    low, high = LIMITS[column]

    assert raw[column].between(low, high).all()


def test_missing_values_only_in_ca_and_thal(raw):
    missing = raw.isna().sum()

    assert missing[missing > 0].to_dict() == {"ca": 4, "thal": 2}
