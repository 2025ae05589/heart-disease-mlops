"""Data quality checks on the raw Cleveland file.

The number of patients and columns, duplicate rows, valid codes, impossible
numerical values and the places of the missing values are checked. They
repeat the checks of the EDA notebook (section 5), so that the pipeline
fails if the data changes.
"""
import pandas as pd
import pytest

from src.config import ABOVE_ZERO, RAW_PATH, ROOT, VALID_CODES, ZERO_OR_ABOVE

# the coded features, and the diagnosis column of the raw file
CODES = dict(VALID_CODES)
CODES["num"] = [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def raw():
    """The raw Cleveland file, read once for all the tests."""
    return pd.read_csv(ROOT / RAW_PATH)


def test_number_of_patients_and_columns(raw):
    assert raw.shape == (303, 14)
    assert set(raw.columns) == set(CODES) | set(ABOVE_ZERO) | set(ZERO_OR_ABOVE)


def test_no_duplicate_rows(raw):
    assert raw.duplicated().sum() == 0


@pytest.mark.parametrize("column", list(CODES))
def test_codes_are_valid(raw, column):
    # missing values are checked separately
    values = raw[column].dropna()

    assert values.isin(CODES[column]).all()


@pytest.mark.parametrize("column", ABOVE_ZERO + ZERO_OR_ABOVE)
def test_numerical_values_are_possible(raw, column):
    if column in ABOVE_ZERO:
        assert (raw[column] > 0).all()
    else:
        assert (raw[column] >= 0).all()


def test_missing_values_only_in_ca_and_thal(raw):
    missing = raw.isna().sum()

    assert missing[missing > 0].to_dict() == {"ca": 4, "thal": 2}
