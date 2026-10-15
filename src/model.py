"""PyTorch model for Titanic survival: definition, training, prediction, saving and loading.

Used by ``train.py`` (to train and save the model) and by the Streamlit app
(to load the model and predict).

The trained model is saved as one file (default ``models/titanic_mlp.pt``) that
holds everything needed for prediction:

* ``model_state``: the network weights.
* ``architecture``: the number of inputs, the hidden layer sizes and the dropout,
  so the network can be rebuilt before the weights are loaded.
* ``preprocessor``: the values the preprocessor learned from the training data
  (fill values, categories, scaling statistics, feature names).
* ``threshold``: the classification threshold chosen on the validation set
  (survived when ``probability >= threshold``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import TitanicPreprocessor

# Training and prediction run on CPU only.
DEVICE = torch.device("cpu")

# Network architecture: one hidden layer of 32 units, dropout 0.2.
HIDDEN_SIZES = [32]
DROPOUT = 0.2


class MLP(nn.Module):
    """Multilayer perceptron for binary classification.

    Each hidden layer is ``Linear -> ReLU -> Dropout``. A final ``Linear`` layer
    gives one output per row. The model returns raw logits (no sigmoid), which is
    what ``nn.BCEWithLogitsLoss`` expects.
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int] = HIDDEN_SIZES, dropout: float = DROPOUT) -> None:
        """Build the layers.

        Args:
            input_dim: Number of input features.
            hidden_sizes: Number of units in each hidden layer.
            dropout: Dropout probability after each hidden layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = dropout

        layers: list[nn.Module] = []
        previous_size = input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(previous_size, size), nn.ReLU(), nn.Dropout(dropout)]
            previous_size = size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``.

        Args:
            x: Float tensor of shape ``(batch, input_dim)``.
        """
        return self.network(x).squeeze(-1)


def to_tensor(values: pd.DataFrame | pd.Series) -> torch.Tensor:
    """Convert a DataFrame or Series to a float32 tensor on ``DEVICE``."""
    return torch.tensor(values.to_numpy(dtype=np.float32), device=DEVICE)


def train_model(
    model: MLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> dict:
    """Train the model with Adam and early stopping on the validation loss.

    Each epoch makes one pass over shuffled mini-batches of the training data,
    then computes the loss on the whole validation data. The weights with the
    lowest validation loss are kept and loaded back into the model at the end.

    Args:
        model: The model to train (modified in place).
        X_train: Training features, float32, shape ``(rows, features)``.
        y_train: Training targets, float32 0/1, shape ``(rows,)``.
        X_val: Validation features, float32.
        y_val: Validation targets, float32 0/1.
        learning_rate: Adam learning rate.
        batch_size: Rows per mini-batch.
        max_epochs: Maximum number of passes over the training data.
        patience: Stop after this many epochs without a lower validation loss.
        seed: Seed for the mini-batch shuffling.

    Returns:
        A history dict with ``train_loss`` and ``val_loss`` (one value per epoch),
        ``best_epoch``, ``best_val_loss`` and ``stopped_epoch``. Epochs are 1-based.
        The model is left with the weights of ``best_epoch``, in eval mode.
    """
    # A seeded generator makes the batch order the same on every run.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    history: dict = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        # 1. One pass over the training batches.
        model.train()
        batch_losses = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        history["train_loss"].append(float(np.mean(batch_losses)))

        # 2. Loss on the whole validation set.
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history["val_loss"].append(val_loss)

        # 3. Early stopping: keep the weights of the best epoch so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_val_loss=best_val_loss, stopped_epoch=epoch)
    return history


def predict_proba(model: MLP, preprocessor: TitanicPreprocessor, df: pd.DataFrame) -> np.ndarray:
    """Return the survival probability of each row of ``df``.

    This is the only prediction path in the project.

    Args:
        model: A trained model.
        preprocessor: The preprocessor fitted together with the model.
        df: Raw Titanic rows. ``Survived`` may be present or not; it is ignored.

    Returns:
        A 1-D numpy array of probabilities between 0 and 1, one per row.
    """
    X = to_tensor(preprocessor.transform(df))
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
    return probabilities.cpu().numpy()


def save_model(model: MLP, preprocessor: TitanicPreprocessor, threshold: float, path: str | Path) -> None:
    """Save the model weights, architecture, preprocessor and classification threshold to one file.

    Args:
        model: The trained model.
        preprocessor: The preprocessor fitted together with the model.
        threshold: The classification threshold chosen on the validation set
            (survived when ``probability >= threshold``).
        path: File to write (for example ``models/titanic_mlp.pt``); its folder is created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "architecture": {"input_dim": model.input_dim, "hidden_sizes": model.hidden_sizes, "dropout": model.dropout},
        "preprocessor": preprocessor.to_dict(),
        "threshold": float(threshold),
    }
    torch.save(checkpoint, path)


def load_model(path: str | Path) -> tuple[MLP, TitanicPreprocessor, float]:
    """Load a model, its preprocessor and its classification threshold saved with ``save_model``.

    Args:
        path: The saved model file.

    Returns:
        A tuple ``(model, preprocessor, threshold)``, ready for ``predict_proba``.
        The model is in eval mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no classification threshold (saved by an older version).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run `python train.py` first.")

    # weights_only=True loads only tensors and plain Python values, never arbitrary code.
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    if "threshold" not in checkpoint:
        raise ValueError("The model file has no classification threshold. Run python train.py to retrain the model.")

    model = MLP(**checkpoint["architecture"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocessor = TitanicPreprocessor.from_dict(checkpoint["preprocessor"])
    return model, preprocessor, checkpoint["threshold"]
