"""Download the UCI Heart Disease data and save it as raw CSV files.

Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease

Run from the project root:
    python scripts/download_data.py

Output (one file per hospital):
    data/raw/heart_cleveland_raw.csv     <- used for modelling
    data/raw/heart_hungarian_raw.csv     <- used only in the EDA comparison
    data/raw/heart_switzerland_raw.csv   <- used only in the EDA comparison
    data/raw/heart_va_raw.csv            <- used only in the EDA comparison
"""
from pathlib import Path

import pandas as pd

# Direct folder of the data files on the UCI site. The dataset page above
# gives the same files inside a zip.
BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
)

# The dataset has records from four hospitals. Only the "processed" files are
# downloaded. They have the 14 columns used in all published work and carry
# no patient names or ID numbers.
#
# Cleveland is the file used for modelling, for two reasons:
#   1. The UCI page says "the Cleveland database is the only one that has
#      been used by ML researchers to date".
#   2. It is almost complete (297 of 303 rows have no missing value). The
#      other three files have at most 1 complete row each.
#
# The number against each file is the row count in the UCI documentation.
HOSPITALS = {
    "cleveland": ("processed.cleveland.data", 303),
    "hungarian": ("processed.hungarian.data", 294),
    "switzerland": ("processed.switzerland.data", 123),
    "va": ("processed.va.data", 200),
}

# The source files have no header row, so the names are given here.
# They follow the order in the UCI documentation (heart-disease.names).
COLUMNS = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1 to 4)
    "trestbps",  # resting blood pressure in mm Hg
    "chol",      # serum cholesterol in mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)
    "restecg",   # resting ECG result (0 to 2)
    "thalach",   # maximum heart rate reached
    "exang",     # chest pain during exercise (1 = yes, 0 = no)
    "oldpeak",   # ST depression caused by exercise
    "slope",     # slope of the ST segment at peak exercise (1 to 3)
    "ca",        # number of major vessels seen in fluoroscopy (0 to 3)
    "thal",      # thallium test result (3, 6 or 7)
    "num",       # diagnosis: 0 = no disease, 1 to 4 = disease
]

RAW_DIR = Path("data/raw")


def raw_path(hospital: str) -> Path:
    """Return the CSV path for one hospital, e.g. heart_cleveland_raw.csv."""
    return RAW_DIR / f"heart_{hospital}_raw.csv"


def download(hospital: str) -> pd.DataFrame:
    """Read one hospital file from UCI, add column names and save it as a CSV.

    The file is saved without any cleaning, so that the cleaning step can
    always be repeated from the original data.
    """
    file_name, expected_rows = HOSPITALS[hospital]

    # Missing values are written as "?" in the source files.
    df = pd.read_csv(
        BASE_URL + file_name, header=None, names=COLUMNS, na_values="?"
    )

    # Stop early if UCI changes a file, instead of failing later in training.
    expected_shape = (expected_rows, len(COLUMNS))
    if df.shape != expected_shape:
        raise ValueError(
            f"{file_name}: expected shape {expected_shape}, got {df.shape}"
        )

    out_path = raw_path(hospital)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    for name in HOSPITALS:
        data = download(name)
        complete_rows = len(data.dropna())
        print(
            f"{name:12s} rows={len(data):4d}  complete rows={complete_rows:4d}"
            f"  saved to {raw_path(name)}"
        )
