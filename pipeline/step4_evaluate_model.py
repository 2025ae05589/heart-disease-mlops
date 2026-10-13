"""Step 4: select one model, evaluate it once on the test patients and apply
gate 1. The step ends with an error if the gate fails.

The models are read from the MLflow runs of step 3. The test results are
added to the run of the selected model, where step 5 finds them.

Run from the project root, after step 3:
    python -m pipeline.step4_evaluate_model
"""
import logging
import sys

from src.config import LOG_DATE_FORMAT, LOG_FORMAT, ROC_AUC_GATE
from src.data.preparation import split_data
from src.models.evaluation import compute_metrics, passes_quality_gate, select_best_model
from src.tracking.experiments import load_model_results, load_trained_model, log_test_results

log = logging.getLogger("step4_evaluate_model")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # the latest parent run of each model, as logged by step 3
    selected = select_best_model(load_model_results())
    log.info("Selected model: %s (%s)", selected["model"], selected["best_setting"])

    # the test set is used once, for the selected model only
    _, X_test, _, y_test = split_data()
    model = load_trained_model(selected["run_id"])
    probability = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, probability)
    log_test_results(selected["run_id"], selected["model"], test_metrics, y_test, probability)
    roc_auc = test_metrics["roc_auc"]
    log.info("Test set: ROC-AUC %.3f, recall %.3f", roc_auc, test_metrics["recall"])

    # gate 1: lowest ROC-AUC accepted on the test set
    if not passes_quality_gate(test_metrics):
        log.error("Gate 1 failed: test ROC-AUC %.3f is below %.2f", roc_auc, ROC_AUC_GATE)
        sys.exit(1)
    log.info("Gate 1 passed: test ROC-AUC %.3f is at least %.2f", roc_auc, ROC_AUC_GATE)


if __name__ == "__main__":
    main()
