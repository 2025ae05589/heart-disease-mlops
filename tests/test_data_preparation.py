"""Tests for src/data/preparation.py: the cleaning and the split into
training and test patients.

The cleaning is tested on three made-up rows and on the files in the
repository. The split is tested for its sizes, for the same share of disease
in both parts, and for giving the same patients every time.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL, CLEAN_PATH, FEATURES, NUMERIC, RAW_PATH, ROOT, TARGET
from src.data.preparation import clean, prepare, split_data


def raw_sample() -> pd.DataFrame:
    """Three raw rows: no disease, disease, and disease with "ca" missing."""
    return pd.DataFrame({
        "age": [63.0, 67.0, 41.0],
        "sex": [1.0, 1.0, 0.0],
        "cp": [1.0, 4.0, 2.0],
        "trestbps": [145.0, 160.0, 130.0],
        "chol": [233.0, 286.0, 204.0],
        "fbs": [1.0, 0.0, 0.0],
        "restecg": [2.0, 2.0, 2.0],
        "thalach": [150.0, 108.0, 172.0],
        "exang": [0.0, 1.0, 0.0],
        "oldpeak": [2.3, 1.5, 1.4],
        "slope": [3.0, 2.0, 1.0],
        "ca": [0.0, 3.0, np.nan],
        "thal": [6.0, 3.0, 3.0],
        "num": [0, 2, 4],
    })


def test_target_is_binary():
    cleaned = clean(raw_sample())

    assert cleaned[TARGET].tolist() == [0, 1, 1]
    assert "num" not in cleaned.columns


def test_columns_and_their_order():
    cleaned = clean(raw_sample())

    assert cleaned.columns.tolist() == NUMERIC + CATEGORICAL + [TARGET]


def test_missing_value_is_kept():
    cleaned = clean(raw_sample())

    assert len(cleaned) == 3
    assert cleaned["ca"].isna().tolist() == [False, False, True]


def test_codes_are_whole_numbers_when_nothing_is_missing():
    cleaned = clean(raw_sample())

    assert pd.api.types.is_integer_dtype(cleaned["cp"])
    # "ca" has a missing value, so it cannot be stored as int
    assert pd.api.types.is_float_dtype(cleaned["ca"])


def test_input_is_not_changed():
    raw = raw_sample()
    clean(raw)

    assert "num" in raw.columns
    assert TARGET not in raw.columns


def test_prepare_saves_the_cleaned_file(tmp_path):
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "processed" / "clean.csv"
    raw_sample().to_csv(raw_path, index=False)

    cleaned = prepare(raw_path, clean_path)

    saved = pd.read_csv(clean_path)
    assert saved.shape == cleaned.shape == (3, 14)
    assert saved[TARGET].tolist() == [0, 1, 1]


def test_clean_file_in_the_repository():
    data = pd.read_csv(ROOT / CLEAN_PATH)

    assert data.shape == (303, 14)
    assert set(data[TARGET]) == {0, 1}
    assert int(data[TARGET].sum()) == 139
    # only "ca" and "thal" have missing values
    missing = data.isna().sum()
    assert missing[missing > 0].to_dict() == {"ca": 4, "thal": 2}


def test_clean_file_comes_from_the_raw_file():
    cleaned = clean(pd.read_csv(ROOT / RAW_PATH))
    saved = pd.read_csv(ROOT / CLEAN_PATH)

    pd.testing.assert_frame_equal(cleaned, saved, check_dtype=False)


def test_split_sizes():
    X_train, X_test, y_train, y_test = split_data()

    assert len(X_train) == len(y_train) == 242
    assert len(X_test) == len(y_test) == 61
    assert X_train.columns.tolist() == FEATURES


def test_split_is_stratified():
    _, _, y_train, y_test = split_data()

    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_split_is_the_same_every_time():
    first = split_data()
    second = split_data()

    assert first[1].index.tolist() == second[1].index.tolist()


def test_no_patient_is_in_both_parts():
    X_train, X_test, _, _ = split_data()

    assert set(X_train.index).isdisjoint(X_test.index)
