"""Tests for the data split, the folds, the selection rule, the metrics and
the quality gate."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    CV_FOLDS,
    FEATURES,
    ROC_AUC_GATE,
    compute_metrics,
    create_cv_folds,
    passes_quality_gate,
    select_best_model,
    split_data,
)


def test_split_sizes():
    X_train, X_test, y_train, y_test = split_data()

    assert len(X_train) == len(y_train) == 242
    assert len(X_test) == len(y_test) == 61
    assert X_train.columns.tolist() == FEATURES


def test_split_is_stratified():
    _, _, y_train, y_test = split_data()

    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_split_is_the_same_every_time():
    first = split_data()
    second = split_data()

    assert first[1].index.tolist() == second[1].index.tolist()


def test_no_patient_is_in_both_parts():
    X_train, X_test, _, _ = split_data()

    assert set(X_train.index).isdisjoint(X_test.index)


def test_each_training_patient_is_validated_once():
    X_train, _, y_train, _ = split_data()
    folds = create_cv_folds()

    validated = []
    for _, validation_rows in folds.split(X_train, y_train):
        validated.extend(validation_rows)

    assert folds.get_n_splits() == CV_FOLDS
    assert sorted(validated) == list(range(len(X_train)))


def results_table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "roc_auc_mean", "recall_mean"])


def test_highest_recall_among_the_close_models_is_selected():
    results = results_table([
        ("A", 0.905, 0.78),
        ("B", 0.903, 0.80),
        ("C", 0.899, 0.76),
    ])

    assert select_best_model(results)["model"] == "B"


def test_model_far_from_the_best_roc_auc_is_not_selected():
    results = results_table([
        ("A", 0.905, 0.78),
        # highest recall, but its ROC-AUC is not close to the best
        ("B", 0.850, 0.95),
    ])

    assert select_best_model(results)["model"] == "A"


def test_metrics_on_a_worked_example():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    probability = np.array([0.9, 0.8, 0.4, 0.6, 0.2, 0.1])

    metrics = compute_metrics(y_true, probability)

    # at the 0.5 cut-off: 2 found, 1 missed, 1 false alarm, 2 correctly cleared
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    # 8 of the 9 (disease, no disease) pairs are ranked correctly
    assert metrics["roc_auc"] == pytest.approx(8 / 9)


def test_probability_at_the_cut_off_counts_as_disease():
    metrics = compute_metrics(np.array([1, 0]), np.array([0.5, 0.2]))

    assert metrics["recall"] == 1.0


def test_quality_gate():
    assert passes_quality_gate({"roc_auc": ROC_AUC_GATE})
    assert passes_quality_gate({"roc_auc": 0.95})
    assert not passes_quality_gate({"roc_auc": ROC_AUC_GATE - 0.001})
