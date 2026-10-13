"""Tests for src/models/evaluation.py: the selection rule, the metrics and
the quality gate.

Small made-up tables and labels are used, for which the right answer can be
worked out by hand. No model is trained here.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import ROC_AUC_GATE
from src.models.evaluation import compute_metrics, passes_quality_gate, select_best_model


def results_table(rows) -> pd.DataFrame:
    """A small table of models, as given to select_best_model()."""
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
