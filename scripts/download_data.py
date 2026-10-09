"""Download the UCI Heart Disease data.

Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease

Run from the project root:
    python scripts/download_data.py

For each hospital, the original UCI file and a CSV with column names are
saved in data/raw/.
"""
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

# folder of the data files on the UCI site
BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
)

# One file per hospital, with its row count from the UCI documentation and
# the SHA-256 hash of the file. Only Cleveland is used for modelling.
HOSPITALS = {
    "cleveland": (
        "processed.cleveland.data", 303,
        "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    ),
    "hungarian": (
        "processed.hungarian.data", 294,
        "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    ),
    "switzerland": (
        "processed.switzerland.data", 123,
        "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    ),
    "va": (
        "processed.va.data", 200,
        "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
    ),
}

# The source files have no header row. The names and meanings are from the
# UCI documentation (heart-disease.names).
COLUMNS = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1 to 4)
    "trestbps",  # resting blood pressure in mm Hg
    "chol",      # serum cholesterol in mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # resting electrocardiographic results (0 to 2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of the peak exercise ST segment (1 to 3)
    "ca",        # number of major vessels (0 to 3) colored by fluoroscopy
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # diagnosis: 0 = no presence, 1 to 4 = presence
]

RAW_DIR = Path("data/raw")


def raw_path(hospital: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the CSV path for one hospital, e.g. heart_cleveland_raw.csv."""
    return Path(raw_dir) / f"heart_{hospital}_raw.csv"


def source_path(hospital: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the path of the original UCI file for one hospital."""
    return Path(raw_dir) / HOSPITALS[hospital][0]


def download(hospital: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download one hospital file and save it as it is, and as a CSV with
    column names. No cleaning is done here."""
    file_name, expected_rows, expected_hash = HOSPITALS[hospital]

    with urllib.request.urlopen(BASE_URL + file_name) as response:
        content = response.read()

    # stop if the file on the UCI site has changed
    found_hash = hashlib.sha256(content).hexdigest()
    if found_hash != expected_hash:
        raise ValueError(
            f"{file_name}: expected hash {expected_hash}, got {found_hash}"
        )

    source = source_path(hospital, raw_dir)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)

    # missing values are written as "?" in the source files
    df = pd.read_csv(
        io.BytesIO(content), header=None, names=COLUMNS, na_values="?"
    )

    expected_shape = (expected_rows, len(COLUMNS))
    if df.shape != expected_shape:
        raise ValueError(
            f"{file_name}: expected shape {expected_shape}, got {df.shape}"
        )

    df.to_csv(raw_path(hospital, raw_dir), index=False)
    return df


if __name__ == "__main__":
    for name in HOSPITALS:
        data = download(name)
        complete_rows = len(data.dropna())
        print(
            f"{name:12s} rows={len(data):4d}  complete rows={complete_rows:4d}"
            f"  saved to {raw_path(name)}"
        )
