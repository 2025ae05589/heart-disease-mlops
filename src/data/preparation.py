"""Preparation of the Cleveland data: cleaning, and the split into training
and test patients.

Cleaning makes the target binary and puts the columns in a fixed order.
Missing values are left for the model pipeline. The split is stratified with
a fixed seed, so every step and notebook gets the same patients.

Used by steps 2, 3 and 4 of the pipeline, the notebooks and the tests.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL,
    CLEAN_PATH,
    FEATURES,
    NUMERIC,
    RANDOM_STATE,
    RAW_PATH,
    ROOT,
    TARGET,
    TEST_SIZE,
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Cleveland data: binary target, codes
    as whole numbers and columns in a fixed order. Missing values are left
    as they are, to be filled in the model pipeline."""
    df = df.copy()

    # binary target: "num" is 0 for no disease and 1 to 4 for disease
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # codes are stored as int, except in a column with a missing value (NaN)
    for col in CATEGORICAL:
        if df[col].notna().all():
            df[col] = df[col].astype(int)

    return df[NUMERIC + CATEGORICAL + [TARGET]]


def prepare(raw_path: Path = ROOT / RAW_PATH,
            clean_path: Path = ROOT / CLEAN_PATH) -> pd.DataFrame:
    """Read the raw CSV, clean it and save the result as the cleaned CSV.
    Returns the cleaned data."""
    cleaned = clean(pd.read_csv(raw_path))
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    return cleaned


def split_data():
    """Read the cleaned file and return X_train, X_test, y_train, y_test.

    The split is 80/20, stratified on the target, with a fixed seed. So it
    gives the same patients on every call.
    """
    data = pd.read_csv(ROOT / CLEAN_PATH)

    # float, because an int column cannot hold a missing value (NaN),
    # and new patient data may have one
    X = data[FEATURES].astype(float)
    y = data[TARGET]

    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
