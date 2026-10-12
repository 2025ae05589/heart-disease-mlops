"""Train the models, select one, and make it the champion if it passes the
gates.

Run from the project root:
    python -m src.train
"""
import logging
import sys

from src.evaluation import (
    ROC_AUC_GATE,
    build_metrics_table,
    compute_metrics,
    create_cv_folds,
    passes_quality_gate,
    run_grid_search,
    select_best_model,
    split_data,
)
from src.models import MODELS
from src.preprocessing import make_pipeline
from src.tracking import (
    EXPORT_DIR,
    MODEL_NAME,
    champion_test_roc_auc,
    export_champion,
    load_model_results,
    load_trained_model,
    log_model_runs,
    log_test_results,
    mark_gate_failed,
    promote_to_champion,
    register_challenger,
)

log = logging.getLogger("train")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    X_train, X_test, y_train, y_test = split_data()
    folds = create_cv_folds()
    log.info("Data: %d training patients, %d test patients", len(X_train), len(X_test))

    # tune each model with cross-validation on the training patients
    for model_name, (model, grid) in MODELS.items():
        pipeline = make_pipeline(model)
        search = run_grid_search(pipeline, grid, X_train, y_train, folds)
        metrics_table = build_metrics_table(search)
        log_model_runs(model_name, pipeline, search, metrics_table, X_train, y_train, folds)
        log.info("%s: cross-validation ROC-AUC %.3f", model_name, search.best_score_)

    # the latest parent run of each model is the one logged above
    selected = select_best_model(load_model_results())
    log.info("Selected model: %s (%s)", selected["model"], selected["best_setting"])

    # the test set is used once, for the selected model only
    model = load_trained_model(selected["run_id"])
    probability = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, probability)
    log_test_results(selected["run_id"], selected["model"], test_metrics, y_test, probability)
    roc_auc = test_metrics["roc_auc"]
    log.info("Test set: ROC-AUC %.3f, recall %.3f", roc_auc, test_metrics["recall"])

    number = register_challenger(selected["run_id"], test_metrics)
    log.info("Registered %s version %s as challenger", MODEL_NAME, number)

    # gate 1: lowest ROC-AUC accepted on the test set
    if not passes_quality_gate(test_metrics):
        mark_gate_failed(number, "gate 1")
        log.error("Gate 1 failed: test ROC-AUC %.3f is below %.2f", roc_auc, ROC_AUC_GATE)
        sys.exit(1)
    log.info("Gate 1 passed: test ROC-AUC %.3f is at least %.2f", roc_auc, ROC_AUC_GATE)

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

    exported = export_champion()
    log.info("Champion (version %s) exported to %s", exported, EXPORT_DIR)


if __name__ == "__main__":
    main()
