"""Train the Titanic survival model and save it to disk.

Steps:
    1. Load the full dataset (Kaggle's train.csv). If data/train.csv is missing,
       it is downloaded from Kaggle first (see src/data_download.py for the
       Kaggle credentials this needs).
    2. Hold out the test set with the shared split function. The test set is not
       used here; it is kept for evaluation in the Streamlit app.
    3. Split the training set into train_set (fitting) and validation_set
       (early stopping only).
    4. Fit the preprocessor on train_set and transform both sets.
    5. Train the MLP with early stopping.
    6. Choose the classification threshold on validation_set (highest accuracy).
    7. Save the model (weights, architecture, preprocessor and threshold) to one file.
    8. Write the test set to a CSV file, for evaluation in the Streamlit app.

Usage:
    python train.py
    python train.py --data data/train.csv --model-path models/titanic_mlp.pt

Writes:
    data/train.csv          downloaded from Kaggle, only if it is not there yet
    models/titanic_mlp.pt   the trained model (or the path given with --model-path)
    data/test_set.csv       the held-out test set, with all original columns including Survived
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data_download import DatasetDownloadError, ensure_train_csv
from src.data_split import split_dataset
from src.evaluation import compute_metrics, select_threshold
from src.model import DROPOUT, HIDDEN_SIZES, MLP, predict_proba, save_model, to_tensor, train_model
from src.preprocessing import TitanicPreprocessor, split_features_target

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 2024
TEST_SIZE = 0.2          # share of full_df held out as the test set
VALIDATION_SIZE = 0.2    # share of train_df used as the validation set

LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 20            # epochs without improvement before early stopping

DEFAULT_DATA_PATH = Path("data/train.csv")
DEFAULT_MODEL_PATH = Path("models/titanic_mlp.pt")
TEST_SET_PATH = Path("data/test_set.csv")
TARGET = "Survived"

logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch so every run gives the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: Path) -> pd.DataFrame:
    """Load the full dataset (Kaggle's train.csv), downloading it from Kaggle if it is missing.

    Args:
        path: Where train.csv is (or will be saved after the download).

    Returns:
        full_df: all rows of the file.
    """
    try:
        ensure_train_csv(path)
    except DatasetDownloadError as error:
        sys.exit(str(error))
    return pd.read_csv(path)


def fit_and_train(
    train_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    hidden_sizes: list[int] = HIDDEN_SIZES,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    seed: int = SEED,
) -> tuple[MLP, TitanicPreprocessor, dict]:
    """Go from two DataFrames to a trained model: preprocess, build the MLP, train it.

    This is the single training unit of the project. ``main()`` calls it once with the
    default settings; a hyperparameter-search script can call it with other settings
    on other splits, and runs exactly the same code.

    Args:
        train_set: Rows the preprocessor and the model are fitted on (with ``Survived``).
        validation_set: Rows used only for early stopping (with ``Survived``).
        hidden_sizes: Number of units in each hidden layer.
        dropout: Dropout probability after each hidden layer.
        learning_rate: Adam learning rate.
        batch_size: Rows per mini-batch.
        max_epochs: Maximum number of passes over the training data.
        patience: Epochs without improvement before early stopping.
        seed: Seed for the weight initialisation, dropout and batch shuffling.

    Returns:
        A tuple ``(model, preprocessor, history)``: the trained model (weights of the
        best epoch, eval mode), the preprocessor fitted on ``train_set``, and the
        training history from ``train_model``.
    """
    # Seed first, so the weight initialisation below is the same on every call.
    set_seed(seed)

    # Fit the preprocessor on train_set only, then transform both sets.
    train_inputs, train_target = split_features_target(train_set)
    validation_inputs, validation_target = split_features_target(validation_set)
    preprocessor = TitanicPreprocessor().fit(train_inputs)
    X_train = to_tensor(preprocessor.transform(train_inputs))
    X_val = to_tensor(preprocessor.transform(validation_inputs))
    y_train, y_val = to_tensor(train_target), to_tensor(validation_target)

    # Build the model (the weights are initialised here) and train it.
    model = MLP(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        learning_rate=learning_rate, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, seed=seed,
    )
    return model, preprocessor, history


def main() -> None:
    """Run the full training process: load, split, preprocess, train, choose the threshold, save."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the Titanic survival MLP and save it to disk.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to Kaggle's train.csv.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model.")
    args = parser.parse_args()

    # 1. Load the data.
    full_df = load_data(args.data)
    logger.info("Loaded full_df from %s: %d rows", args.data, len(full_df))

    # 2. Hold out the test set. test_df is not used for training; it is only
    #    written to disk in step 8, for evaluation in the app.
    train_df, test_df = split_dataset(full_df, test_size=TEST_SIZE, seed=SEED)
    logger.info("Held out the test set: train_df=%d rows, test_df=%d rows", len(train_df), len(test_df))

    # 3. Split the training set (train_df, 80% of full_df) into
    #    train_set (fitting the preprocessor and the model) and validation_set (early stopping only).
    train_set, validation_set = train_test_split(
        train_df, test_size=VALIDATION_SIZE, stratify=train_df[TARGET], random_state=SEED, shuffle=True
    )
    logger.info("Split train_df: train_set=%d rows, validation_set=%d rows", len(train_set), len(validation_set))

    # 4–5. Fit the preprocessor on train_set, build the MLP and train it (see fit_and_train).
    logger.info("Preprocessing and training the model...")
    model, preprocessor, history = fit_and_train(train_set, validation_set)
    logger.info("Preprocessed the data: %d features", len(preprocessor.feature_names))
    logger.info(
        "Training finished: best epoch=%d, stopped at epoch=%d, best validation loss=%.4f",
        history["best_epoch"], history["stopped_epoch"], history["best_val_loss"],
    )

    # 6. Choose the classification threshold on validation_set.
    validation_probabilities = predict_proba(model, preprocessor, validation_set)
    threshold = select_threshold(validation_set[TARGET], validation_probabilities)
    validation_accuracy = compute_metrics(validation_set[TARGET], validation_probabilities, threshold)["accuracy"]
    logger.info("Chose the threshold on validation_set: threshold=%.4f, validation accuracy=%.4f", threshold, validation_accuracy)

    # 7. Save the model, the preprocessor and the threshold to one file.
    save_model(model, preprocessor, threshold, args.model_path)
    logger.info("Saved the model to %s", args.model_path)

    # 8. Write the test set for evaluation in the app. train.py does not use it in any other way.
    TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(TEST_SET_PATH, index=False)
    logger.info("Wrote the test set to %s: %d rows", TEST_SET_PATH, len(test_df))


if __name__ == "__main__":
    main()
