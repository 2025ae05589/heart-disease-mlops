"""Step 2: clean the raw Cleveland data and save the clean file.

Reads data/raw/heart_cleveland_raw.csv and writes
data/processed/heart_clean.csv, the file used for training.

Run from the project root:
    python -m pipeline.step2_prepare_data
"""
import logging

from src.config import CLEAN_PATH, LOG_DATE_FORMAT, LOG_FORMAT, RAW_PATH, ROOT, TARGET
from src.data.preparation import prepare

log = logging.getLogger("step2_prepare_data")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    data = prepare(ROOT / RAW_PATH, ROOT / CLEAN_PATH)
    with_disease = int(data[TARGET].sum())

    log.info("Clean data saved to %s", CLEAN_PATH.as_posix())
    log.info("%d patients, %d columns", data.shape[0], data.shape[1])
    log.info("Target: %d without disease, %d with disease", len(data) - with_disease, with_disease)
    log.info("Missing values left for the model pipeline: %d", int(data.isna().sum().sum()))


if __name__ == "__main__":
    main()
