"""Shared train/test split for the Titanic project.

The EDA notebook, the training script and the Streamlit app all import
`split_dataset` from here, so every part of the project uses the same split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Survived"


def split_dataset(
    full_df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the full dataset into a training set and a held-out test set.

    The split is stratified on the target column, so both sets keep the same
    share of survivors. The same inputs always give the same split.

    Args:
        full_df: The full dataset (Kaggle's train.csv, all rows).
        test_size: Share of rows to put in the test set.
        seed: Random seed for the shuffle.

    Returns:
        A tuple (train_df, test_df).
    """
    if TARGET_COLUMN not in full_df.columns:
        raise KeyError(f"Column '{TARGET_COLUMN}' is missing; it is needed to stratify the split.")

    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df[TARGET_COLUMN],
        random_state=seed,
        shuffle=True,
    )
    return train_df, test_df
