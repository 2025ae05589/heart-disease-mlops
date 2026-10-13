"""Step 6: export the champion out of MLflow, for the API and the Docker
image.

The champion is found by its alias and copied to models/champion/. The step
ends with an error if there is no champion.

Run from the project root, after step 5:
    python -m pipeline.step6_export_model
"""
import logging
import sys

from src.config import EXPORT_DIR, LOG_DATE_FORMAT, LOG_FORMAT
from src.deployment.packaging import export_champion
from src.tracking.registry import get_champion

log = logging.getLogger("step6_export_model")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if get_champion() is None:
        log.error("There is no champion to export. Step 5 has to be run first.")
        sys.exit(1)

    number = export_champion()
    log.info("Champion (version %s) exported to %s", number, EXPORT_DIR)


if __name__ == "__main__":
    main()
