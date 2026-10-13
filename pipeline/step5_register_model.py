"""Step 5: register the selected model as the challenger, apply gate 2 and
make it the champion if it passes.

Gate 2: the test ROC-AUC has to be above that of the present champion. It is
skipped if there is no champion yet. A failed gate 2 is a normal end of the
step: the new version stays as challenger and the champion is not changed.

Run from the project root, after step 4:
    python -m pipeline.step5_register_model
"""
import logging
import sys

from src.config import LOG_DATE_FORMAT, LOG_FORMAT, MODEL_NAME
from src.models.evaluation import passes_quality_gate
from src.tracking.experiments import load_selected_model
from src.tracking.registry import (
    champion_test_roc_auc,
    mark_gate_failed,
    promote_to_champion,
    register_challenger,
)

log = logging.getLogger("step5_register_model")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # the model tested in step 4 is found by its tag in MLflow
    selected = load_selected_model()
    if selected is None:
        log.error("No tested model was found. Step 4 has to be run first.")
        sys.exit(1)
    model_name, run_id, test_metrics = selected
    roc_auc = test_metrics["roc_auc"]

    # a model that failed gate 1 is not registered
    if not passes_quality_gate(test_metrics):
        log.error("%s failed gate 1 (test ROC-AUC %.3f)", model_name, roc_auc)
        sys.exit(1)

    number = register_challenger(run_id, test_metrics)
    log.info("Registered %s as %s version %s (challenger)", model_name, MODEL_NAME, number)

    # gate 2: better than the champion, if there is one
    champion_roc_auc = champion_test_roc_auc()
    if champion_roc_auc is None:
        log.info("Gate 2 skipped: there is no champion yet")
        promote_to_champion(number)
        log.info("Version %s is the champion", number)
    elif roc_auc > champion_roc_auc:
        log.info("Gate 2 passed: %.3f is above the champion's %.3f", roc_auc, champion_roc_auc)
        promote_to_champion(number)
        log.info("Version %s is the champion", number)
    else:
        mark_gate_failed(number, "gate 2")
        log.warning(
            "Gate 2 failed: %.3f is not above the champion's %.3f. The champion stays.",
            roc_auc, champion_roc_auc,
        )


if __name__ == "__main__":
    main()
