"""Step 3: tune each model with cross-validation on the training patients and
log the runs to MLflow.

For each of the four models, every combination of its grid is tried. The
parent run of the model holds the best combination and the model trained
with it. The test patients are not used in this step.

Run from the project root:
    python -m pipeline.step3_train_models
"""
import logging

from src.config import LOG_DATE_FORMAT, LOG_FORMAT
from src.data.preparation import split_data
from src.features.preprocessing import make_pipeline
from src.models.definitions import MODELS
from src.models.tuning import build_metrics_table, create_cv_folds, run_grid_search
from src.tracking.experiments import log_model_runs

log = logging.getLogger("step3_train_models")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # the test patients are not used in this step
    X_train, X_test, y_train, _ = split_data()
    folds = create_cv_folds()
    log.info("Data: %d training patients, %d test patients", len(X_train), len(X_test))

    for model_name, (model, grid) in MODELS.items():
        pipeline = make_pipeline(model)
        search = run_grid_search(pipeline, grid, X_train, y_train, folds)
        metrics_table = build_metrics_table(search)
        log_model_runs(model_name, pipeline, search, metrics_table, X_train, y_train, folds)
        log.info("%s: cross-validation ROC-AUC %.3f", model_name, search.best_score_)


if __name__ == "__main__":
    main()
