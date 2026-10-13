"""Run steps 2 to 7 in order: prepare the data, train, evaluate, register,
export and validate. The run stops at the first step that fails.

Step 1, the download, is left out: the data files are kept in the repository.

Run from the project root:
    python -m pipeline.run_all
"""
import logging

from pipeline import (
    step2_prepare_data,
    step3_train_models,
    step4_evaluate_model,
    step5_register_model,
    step6_export_model,
    step7_validate_model,
)

log = logging.getLogger("run_all")

STEPS = [
    step2_prepare_data,
    step3_train_models,
    step4_evaluate_model,
    step5_register_model,
    step6_export_model,
    step7_validate_model,
]


def main():
    for step in STEPS:
        step.main()
        log.info("Finished %s", step.__name__.replace("pipeline.", ""))


if __name__ == "__main__":
    main()
