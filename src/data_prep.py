"""Clean the raw Cleveland heart disease data.

The cleaning steps come from the decisions taken in the notebook
notebooks/01_data_acquisition_and_eda.ipynb. This file only puts those
decisions into code, so that the notebook, the training script and the tests
all use the same cleaning.

Run from the project root:
    python src/data_prep.py

Output:
    data/processed/heart_clean.csv
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_cleveland_raw.csv")
CLEAN_PATH = Path("data/processed/heart_clean.csv")

# Numerical features are quantities. Categorical features are group labels
# stored as number codes. The two groups are kept as lists because the model
# pipeline treats them differently (scaling for numerical, one-hot encoding
# for categorical).
#
# "ca" is a count (0 to 3), so by type it is numerical. It is kept in the
# second list because the disease share does not rise in equal steps with
# the count (section 11 of the EDA notebook), so each count is encoded as
# its own category.
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Cleveland data.

    Steps:
      1. Make the binary target from the diagnosis column "num".
      2. Store categorical features as whole numbers where no value is missing.

    Missing values are NOT filled here. They are filled inside the model
    pipeline, which learns the fill value from the training data only.
    Filling them here would use all rows, including future test rows.
    """
    df = df.copy()  # do not change the table passed in by the caller

    # 1. "num" is 0 for no disease and 1 to 4 for disease.
    #    The original column is dropped so that it cannot be used as a feature.
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # 2. The source file writes codes as decimals (1.0, 0.0). A column with a
    #    missing value has to stay as decimal, because whole-number columns
    #    cannot hold an empty cell.
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
