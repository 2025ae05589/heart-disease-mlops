"""Tests for the data cleaning."""
import numpy as np
import pandas as pd

from src.data_prep import CATEGORICAL, CLEAN_PATH, NUMERIC, RAW_PATH, TARGET, clean, prepare
from src.evaluation import ROOT


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
