"""Evaluation of a model.

The five metrics, the rule that selects one model among those compared, the
two gates (gate 1: lowest test ROC-AUC accepted; gate 2: better than the
champion), and the plots of the ROC curve and the confusion matrix.

Used by steps 4 and 5 of the pipeline, the MLflow logging and the notebook.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from src.config import ROC_AUC_GAP, ROC_AUC_GATE


def select_best_model(results: pd.DataFrame) -> pd.Series:
    """Return the row of the selected model.

    results: one row per model, with its mean ROC-AUC and mean recall from
    cross-validation.

    Rule: among the models within ROC_AUC_GAP of the best mean ROC-AUC, the
    one with the highest mean recall is selected.
    """
    best_roc_auc = results["roc_auc_mean"].max()
    close = results[results["roc_auc_mean"] >= best_roc_auc - ROC_AUC_GAP]
    return close.loc[close["recall_mean"].idxmax()]


def compute_metrics(y_true, probability) -> dict:
    """Return accuracy, precision, recall, F1 and ROC-AUC as a dict.

    probability: predicted probability of disease. 0.5 and above is counted
    as disease.
    """
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction),
        "recall": recall_score(y_true, prediction),
        "f1": f1_score(y_true, prediction),
        "roc_auc": roc_auc_score(y_true, probability),
    }


def passes_quality_gate(test_metrics: dict) -> bool:
    """Return True if the ROC-AUC on the test set is at least ROC_AUC_GATE
    (gate 1)."""
    return test_metrics["roc_auc"] >= ROC_AUC_GATE


def passes_champion_gate(roc_auc: float, champion_roc_auc) -> bool:
    """Return True if a new model may replace the champion (gate 2).

    roc_auc: test ROC-AUC of the new model.
    champion_roc_auc: test ROC-AUC of the present champion, or None if there
        is no champion yet, in which case the gate is passed.

    The new model has to be strictly better, so a model that only equals the
    champion, such as the same model trained again, does not replace it.
    """
    if champion_roc_auc is None:
        return True
    return roc_auc > champion_roc_auc


def cross_val_probabilities(pipeline, X_train, y_train, folds):
    """Return the probability of disease for each training patient, predicted
    by a model that was not trained on that patient (cross-validation)."""
    probabilities = cross_val_predict(
        pipeline, X_train, y_train, cv=folds, method="predict_proba"
    )
    return probabilities[:, 1]


def plot_roc_curve(y_true, probability, title):
    """Return a figure with the ROC curve and the line of random guessing."""
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, probability, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            label="random guessing (AUC = 0.5)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, probability, title):
    """Return a figure with the confusion matrix at the 0.5 cut-off."""
    prediction = (probability >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, prediction, display_labels=["no disease", "disease"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig
