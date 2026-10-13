"""Download of the UCI Heart Disease data.

Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease

The file of each hospital is downloaded, checked against the recorded hash
and size, and saved in data/raw/ in two forms: as published by UCI, and as a
CSV with column names. No cleaning is done here.

Used by step 1 of the pipeline and by the EDA notebook.
"""
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import BASE_URL, COLUMNS, HOSPITALS, RAW_DIR


def raw_path(hospital: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the CSV path for one hospital, e.g. heart_cleveland_raw.csv."""
    return Path(raw_dir) / f"heart_{hospital}_raw.csv"


def source_path(hospital: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the path of the original UCI file for one hospital."""
    return Path(raw_dir) / HOSPITALS[hospital][0]


def download(hospital: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download the file of one hospital, check it and save it in raw_dir.

    The file is saved as it is, and as a CSV with column names. Returns the
    data as a table. A ValueError is raised if the hash, or the number of
    rows and columns, is not the recorded one.
    """
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
