"""Streamlit app: evaluate the trained Titanic survival model on a CSV file.

One flow: enter the path to a CSV file with true labels (``Survived``) -> load the
trained model from disk -> run inference -> show metrics and plots. The default
file is the held-out test set written by ``train.py``.

The app only displays results. Preprocessing and prediction come from
``src/model.py`` (``load_model``, ``predict_proba``); the threshold, metrics and
plots come from ``src/evaluation.py``.

Usage:
    python train.py            # once, to train the model and write data/test_set.csv
    streamlit run ds_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from src.model import MLP, load_model, predict_proba
from src.preprocessing import TitanicPreprocessor

# Relative paths are resolved from the project root (the folder of this file),
# so the app works no matter which folder it is started from.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = "models/titanic_mlp.pt"
DEFAULT_DATA_PATH = "data/test_set.csv"
TARGET = "Survived"

THRESHOLD_METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
THRESHOLD_CAPTION = (
    "Chosen on the validation set during training (highest accuracy). "
    "A passenger is predicted as survived when the predicted probability is at least this value."
)


def fail(message: str) -> NoReturn:
    """Show an error message and stop the app run (no traceback is shown)."""
    st.error(message)
    st.stop()


def resolve_path(path_text: str) -> Path:
    """Turn the text from the input box into a path; relative paths start at the project root."""
    path = Path(path_text.strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_file(path_text: str) -> tuple[Path, pd.DataFrame]:
    """Check the path and read the CSV file.

    Args:
        path_text: The text entered by the user.

    Returns:
        The resolved path and the file's rows.
    """
    if not path_text.strip():
        fail("Enter a path to a CSV file.")
    path = resolve_path(path_text)
    if not path.is_file():
        fail(f"The file was not found: {path}")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        fail(f"The file could not be read as a CSV: {path}")
    if df.empty:
        fail(f"The file could not be read as a CSV (it has no rows): {path}")
    return path, df


def check_labels(df: pd.DataFrame) -> None:
    """Check that the file has a ``Survived`` column with only 0 and 1."""
    if TARGET not in df.columns:
        fail(f"Evaluation needs a `{TARGET}` column with the true labels (0 or 1).")
    labels = df[TARGET]
    if labels.isna().any() or not labels.isin([0, 1]).all():
        fail(f"`{TARGET}` must contain only 0 and 1, with no missing values.")


def load_trained_model() -> tuple[MLP, TitanicPreprocessor, float]:
    """Load the model, its preprocessor and its threshold from ``MODEL_PATH``."""
    try:
        return load_model(PROJECT_ROOT / MODEL_PATH)
    except FileNotFoundError:
        fail(f"No trained model found at `{MODEL_PATH}`. Run `python train.py` first.")
    except ValueError as error:
        fail(str(error))


def show_evaluated_data(path: Path, df: pd.DataFrame) -> None:
    """Section 1: the evaluated file and how many passengers survived or not."""
    st.subheader("Evaluated data")
    st.caption(f"File: {path}")
    total = len(df)
    survived = int((df[TARGET] == 1).sum())
    not_survived = total - survived
    passengers_column, survived_column, not_survived_column = st.columns(3)
    passengers_column.metric("Passengers", total)
    survived_column.metric("Survived", f"{survived} ({survived / total:.1%})")
    not_survived_column.metric("Not survive", f"{not_survived} ({not_survived / total:.1%})")


def show_threshold_free_performance(metrics: dict, figures: dict[str, plt.Figure]) -> None:
    """Section 2: ROC-AUC and PR-AUC, each above its curve."""
    st.subheader("Discrimination")
    roc_column, pr_column = st.columns(2)
    roc_column.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    pr_column.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    roc_column.pyplot(figures["roc"])
    pr_column.pyplot(figures["pr"])


def show_threshold_performance(metrics: dict, figures: dict[str, plt.Figure]) -> None:
    """Section 3: the threshold, then the confusion matrix next to accuracy, precision, recall and F1."""
    st.subheader("Performance at the classification threshold")
    st.metric("Classification threshold", f"{metrics['threshold']:.3f}")
    st.caption(THRESHOLD_CAPTION)

    confusion_column, metrics_column = st.columns(2)
    confusion_column.pyplot(figures["confusion"])
    # The four metrics as a 2 x 2 grid of tiles.
    labels = list(THRESHOLD_METRIC_LABELS.items())
    for row_start in (0, 2):
        for tile, (key, label) in zip(metrics_column.columns(2), labels[row_start:row_start + 2]):
            tile.metric(label, f"{metrics[key]:.3f}")


def show_results(path: Path, df: pd.DataFrame, metrics: dict, figures: dict[str, plt.Figure]) -> None:
    """Show the three result sections, separated by dividers, then close the figures."""
    show_evaluated_data(path, df)
    st.divider()
    show_threshold_free_performance(metrics, figures)
    st.divider()
    show_threshold_performance(metrics, figures)

    for figure in figures.values():
        plt.close(figure)


def main() -> None:
    """Build the page and run the evaluation when the button is clicked."""
    st.set_page_config(page_title="Titanic survival model", layout="wide")
    st.title("Titanic survival model: evaluation")
    st.write("Load the trained model, run it on a CSV file with true labels, and see how well it predicts survival.")

    path_text = st.text_input("Path to a CSV file", value=DEFAULT_DATA_PATH)
    st.caption(f"The default, `{DEFAULT_DATA_PATH}`, is the held-out test set written by `train.py`. "
               "Relative paths start at the project folder.")

    if not st.button("Run evaluation", type="primary"):
        st.stop()

    path, df = read_csv_file(path_text)
    check_labels(df)
    model, preprocessor, threshold = load_trained_model()

    try:
        probabilities = predict_proba(model, preprocessor, df)
    except ValueError as error:  # required feature columns are missing
        fail(str(error))

    try:
        metrics = compute_metrics(df[TARGET], probabilities, threshold)
    except ValueError as error:  # only one class in Survived
        fail(str(error))

    figures = {
        "roc": plot_roc_curve(df[TARGET], probabilities, threshold),
        "pr": plot_pr_curve(df[TARGET], probabilities, threshold),
        "confusion": plot_confusion_matrix(metrics["confusion_matrix"]),
    }
    show_results(path, df, metrics, figures)


if __name__ == "__main__":
    main()
