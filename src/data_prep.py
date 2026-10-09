"""Clean the raw Cleveland heart disease data.

Run from the project root:
    python src/data_prep.py
"""
from pathlib import Path

import pandas as pd

# file as published by UCI
SOURCE_PATH = Path("data/raw/processed.cleveland.data")
# same data with column names added
RAW_PATH = Path("data/raw/heart_cleveland_raw.csv")
# cleaned data, used for training
CLEAN_PATH = Path("data/processed/heart_clean.csv")

# "ca" is a count, but it is treated as categorical (EDA notebook, section 11)
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Cleveland data. Missing values are
    left as they are, to be filled in the model pipeline."""
    df = df.copy()

    # binary target: "num" is 0 for no disease and 1 to 4 for disease
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # codes are stored as int, except in a column with a missing value (NaN)
    for col in CATEGORICAL:
        if df[col].notna().all():
            df[col] = df[col].astype(int)

    return df[NUMERIC + CATEGORICAL + [TARGET]]


def prepare(raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Read the raw CSV, clean it, save the cleaned CSV and return it."""
    cleaned = clean(pd.read_csv(raw_path))
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    return cleaned


if __name__ == "__main__":
    data = prepare()
    print("Saved to:", CLEAN_PATH)
    print("Shape:", data.shape)
    print("Missing values left for the pipeline:", int(data.isna().sum().sum()))
    print("Target counts:")
    print(data[TARGET].value_counts().sort_index())
