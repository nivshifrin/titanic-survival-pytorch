"""Evaluation of the Titanic survival model: threshold, metrics and plots.

This module only calculates: it returns numbers and matplotlib figures and knows
nothing about Streamlit. ``train.py`` uses ``select_threshold`` to choose the
classification threshold on the validation set; the app uses ``compute_metrics``
and the plot functions to show the evaluation of any CSV with true labels.

A passenger is classified as survived when ``probability >= threshold``. This rule
is written in one place only: ``predict_labels``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ArrayLike = np.ndarray | list

CURVE_FIGSIZE = (5, 4.5)
CLASS_LABELS = ["Not survive", "Survived"]
COLOR_MAIN = "#2a78d6"
COLOR_MARKER = "#eb6834"
COLOR_REFERENCE = "#52514e"


def predict_labels(probabilities: ArrayLike, threshold: float) -> np.ndarray:
    """Turn probabilities into 0/1 classes: 1 (survived) where ``probability >= threshold``.

    Args:
        probabilities: Survival probabilities.
        threshold: The classification threshold.

    Returns:
        An integer array of 0s and 1s.
    """
    return (np.asarray(probabilities) >= threshold).astype(int)


def select_threshold(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Choose the threshold with the highest accuracy; on a tie, the one that favours the majority class.

    Candidates are every distinct probability plus 0.5. Each candidate classifies
    with ``predict_labels``, so a candidate equal to a probability puts that row in
    the survived class.

    Tie-break: among the candidates with the highest accuracy, take the one that
    puts the most rows in the majority class of ``y_true``. If the majority class is
    0 (did not survive), that is the highest tied threshold; if it is 1 (survived),
    the lowest. When both classes are equally common, 0 is treated as the majority.

    Args:
        y_true: True 0/1 labels.
        probabilities: Survival probabilities for the same rows.

    Returns:
        The selected threshold.
    """
    y_true = np.asarray(y_true)
    candidates = np.union1d(np.asarray(probabilities, dtype=float), [0.5])
    accuracies = np.array([accuracy_score(y_true, predict_labels(probabilities, c)) for c in candidates])
    best = candidates[accuracies == accuracies.max()]
    # A higher threshold classifies more rows as 0, a lower one more rows as 1.
    majority_class = 1 if y_true.mean() > 0.5 else 0
    return float(best.min() if majority_class == 1 else best.max())


def compute_metrics(y_true: ArrayLike, probabilities: ArrayLike, threshold: float) -> dict:
    """Compute the evaluation metrics.

    Args:
        y_true: True 0/1 labels.
        probabilities: Survival probabilities for the same rows.
        threshold: The classification threshold saved with the model.

    Returns:
        A dict with ``roc_auc`` and ``pr_auc`` (threshold-free); ``accuracy``,
        ``precision``, ``recall`` and ``f1`` at the threshold (positive class =
        survived); ``confusion_matrix`` (rows = true class, columns = predicted
        class, order 0 then 1); ``threshold``; and ``n_samples``.

    Raises:
        ValueError: If ``y_true`` contains only one class.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation needs both classes in Survived; this file contains only one.")

    predicted = predict_labels(probabilities, threshold)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
    }


def _rates_at_threshold(y_true: np.ndarray, probabilities: ArrayLike, threshold: float) -> dict[str, float]:
    """True positive rate, false positive rate and precision at the threshold (for the plot markers)."""
    predicted = predict_labels(probabilities, threshold)
    true_positives = int(((predicted == 1) & (y_true == 1)).sum())
    false_positives = int(((predicted == 1) & (y_true == 0)).sum())
    predicted_positives = true_positives + false_positives
    return {
        "tpr": true_positives / (y_true == 1).sum(),
        "fpr": false_positives / (y_true == 0).sum(),
        "precision": true_positives / predicted_positives if predicted_positives else 1.0,
    }


def plot_roc_curve(y_true: ArrayLike, probabilities: ArrayLike, threshold: float) -> Figure:
    """ROC curve with the diagonal reference line and a marker at the chosen threshold.

    Args:
        y_true: True 0/1 labels.
        probabilities: Survival probabilities.
        threshold: The classification threshold.

    Returns:
        The matplotlib figure.
    """
    y_true = np.asarray(y_true)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    point = _rates_at_threshold(y_true, probabilities, threshold)

    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    ax.plot(fpr, tpr, color=COLOR_MAIN, linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], color=COLOR_REFERENCE, linestyle="--", linewidth=1, label="Random guess")
    ax.scatter(point["fpr"], point["tpr"], s=70, color=COLOR_MARKER, zorder=3, label=f"Threshold {threshold:.3f}")
    ax.set(
        title=f"ROC curve (ROC-AUC = {roc_auc_score(y_true, probabilities):.3f})",
        xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: ArrayLike, probabilities: ArrayLike, threshold: float) -> Figure:
    """Precision-recall curve with the positive-class rate as reference and a marker at the threshold.

    Args:
        y_true: True 0/1 labels.
        probabilities: Survival probabilities.
        threshold: The classification threshold.

    Returns:
        The matplotlib figure.
    """
    y_true = np.asarray(y_true)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    point = _rates_at_threshold(y_true, probabilities, threshold)
    positive_rate = y_true.mean()

    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    ax.plot(recall, precision, color=COLOR_MAIN, linewidth=2, label="Model")
    ax.axhline(positive_rate, color=COLOR_REFERENCE, linestyle="--", linewidth=1, label=f"Survival rate ({positive_rate:.2f})")
    ax.scatter(point["tpr"], point["precision"], s=70, color=COLOR_MARKER, zorder=3, label=f"Threshold {threshold:.3f}")
    ax.set(
        title=f"Precision-recall curve (PR-AUC = {average_precision_score(y_true, probabilities):.3f})",
        xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion: np.ndarray) -> Figure:
    """2x2 heatmap of the confusion matrix with the count in each cell.

    Args:
        confusion: The matrix from ``compute_metrics`` (rows = true class, columns = predicted class).

    Returns:
        The matplotlib figure.
    """
    confusion = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    image = ax.imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            count = confusion[row, column]
            # White text on dark cells, dark text on light cells.
            text_color = "white" if count > confusion.max() / 2 else "black"
            ax.text(column, row, str(count), ha="center", va="center", fontsize=14, color=text_color)
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set(title="Confusion matrix", xlabel="Predicted class", ylabel="True class")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
