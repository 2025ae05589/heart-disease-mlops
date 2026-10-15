"""Step 1: download the four UCI files and check them against the recorded
hashes.

The files are saved in data/raw/. The step ends with an error if a file has
changed or if the site cannot be reached. It is not part of run_all, as the
data files are kept in the repository; it is run by hand and by the weekly
data check.

Run from the project root:
    python -m pipeline.step1_download_data
"""
import logging
import sys

from src.config import HOSPITALS, LOG_DATE_FORMAT, LOG_FORMAT, RAW_DIR, ROOT
from src.data.download import download, raw_path

log = logging.getLogger("step1_download_data")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    for hospital in HOSPITALS:
        try:
            data = download(hospital, ROOT / RAW_DIR)
        except ValueError as error:
            # the file is not the recorded one
            log.error("Source data has changed for %s: %s", hospital, error)
            sys.exit(1)
        except OSError as error:
            # the site cannot be reached
            log.error("Could not download %s: %s", hospital, error)
            sys.exit(1)
        log.info(
            "%s: %d rows, %d complete rows, saved to %s",
            hospital, len(data), len(data.dropna()), raw_path(hospital, RAW_DIR).as_posix(),
        )
    log.info("All %d files match the recorded hashes", len(HOSPITALS))


if __name__ == "__main__":
    main()
