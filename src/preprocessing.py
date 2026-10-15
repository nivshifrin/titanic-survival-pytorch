"""Preprocessing for the Titanic survival model.

This module turns raw Titanic rows (the columns of Kaggle's train.csv) into a fully
numeric feature table with no missing values, ready for the PyTorch MLP.

The steps follow the decisions recorded in eda.ipynb:

==============  ======================================================================
Column          What happens
==============  ======================================================================
PassengerId     Dropped.
Name            Dropped.
Ticket          Dropped.
Cabin           Replaced by ``Cabin_missing`` (1 = missing, 0 = present).
Sex             One-hot encoded.
Embarked        Missing filled with the training-set mode, then one-hot encoded.
Pclass          Numeric, standardized.
SibSp           Numeric, standardized.
Parch           Numeric, standardized.
Age             ``Age_missing`` (0/1) added; missing filled with the training-set
                median; standardized.
Fare            ``log1p`` transform, then standardized.
All inputs      A default fill value learned from the training set is used for any
                missing value at inference time.
==============  ======================================================================

Every learned value (fill values, categories, means and standard deviations) comes
from the data passed to ``fit``, which must be the training set only. The fitted
preprocessor is exported with ``to_dict`` and saved inside the model file, then
rebuilt with ``from_dict``, so the training script and the Streamlit app apply
exactly the same transformation.

Example:
    >>> preprocessor = TitanicPreprocessor().fit(train_df)
    >>> X_train = preprocessor.transform(train_df)
    >>> X_test = preprocessor.transform(test_df)
    >>> state = preprocessor.to_dict()
    >>> same_preprocessor = TitanicPreprocessor.from_dict(state)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

TARGET = "Survived"

# Raw columns that are not used as inputs. They are ignored if present.
DROPPED_COLUMNS = ["PassengerId", "Name", "Ticket"]

# Raw columns read as numbers.
NUMERIC_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Raw columns read as text categories, then one-hot encoded.
CATEGORICAL_COLUMNS = ["Sex", "Embarked"]

# Raw columns that become a 0/1 "is missing" indicator.
# Age keeps its value as well; Cabin is used only through its indicator.
MISSING_INDICATOR_COLUMNS = ["Age", "Cabin"]

# Every raw column the preprocessor needs. A CSV without one of these is rejected.
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["Cabin"]

# Numeric columns that get log1p before scaling.
LOG_COLUMNS = ["Fare"]

# Numeric columns that are standardized (mean 0, standard deviation 1).
# The 0/1 indicators and one-hot columns are left as 0/1.
SCALED_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Version of the saved format (see to_dict). Bump it if the structure changes.
FORMAT_VERSION = 1


class NotFittedError(RuntimeError):
    """Raised when transform or to_dict is called before fit."""


class TitanicPreprocessor:
    """Learn preprocessing values from the training set and apply them to any Titanic data.

    Usage follows the familiar fit / transform pattern:

    * ``fit(train_df)`` learns fill values, category lists and scaling statistics
      from the training set.
    * ``transform(df)`` applies them to any DataFrame with the raw Titanic columns
      (training set, test set, or a new CSV at inference time) and returns a
      float32 feature table with a fixed column order and no missing values.

    Attributes (set by ``fit``):
        fill_values: Default fill value per raw input column. Median for numeric
            columns, mode (most common value) for categorical columns.
        categories: Sorted list of known categories per categorical column. Each
            becomes one one-hot column. A category not seen in training gets 0 in
            every one-hot column of that variable.
        scaling: Mean and standard deviation per scaled column, computed after
            filling and after the log1p transform.
        feature_names: Names of the output columns, in output order.
    """

    def __init__(self) -> None:
        self.fill_values: dict[str, float | str] = {}
        self.categories: dict[str, list[str]] = {}
        self.scaling: dict[str, dict[str, float]] = {}
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once ``fit`` has been called (or a fitted preprocessor was loaded)."""
        return bool(self.feature_names)

    def fit(self, train_df: pd.DataFrame) -> TitanicPreprocessor:
        """Learn all preprocessing values from the training set.

        Args:
            train_df: The training set, with the raw Titanic columns. Only the
                training set may be passed here; never the test set.

        Returns:
            The fitted preprocessor itself, so calls can be chained.
        """
        _check_required_columns(train_df)
        data = _clean_types(train_df)

        # Default fill values: median for numbers, mode for categories.
        self.fill_values = {column: float(data[column].median()) for column in NUMERIC_COLUMNS}
        for column in CATEGORICAL_COLUMNS:
            self.fill_values[column] = str(data[column].mode().iloc[0])

        filled = self._fill_missing(data)

        # Known categories, sorted so the one-hot column order is stable.
        self.categories = {column: sorted(filled[column].unique().tolist()) for column in CATEGORICAL_COLUMNS}

        # Scaling statistics, computed on the same values transform() will scale.
        logged = self._apply_log(filled)
        self.scaling = {}
        for column in SCALED_COLUMNS:
            std = float(logged[column].std(ddof=0))
            self.scaling[column] = {
                "mean": float(logged[column].mean()),
                # A constant column has std 0; use 1 so scaling does not divide by zero.
                "std": std if std > 0 else 1.0,
            }

        self.feature_names = self._build_feature_names()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn raw Titanic rows into the model's feature table.

        Steps, in order:
            1. Check that the required raw columns exist.
            2. Convert column types (numbers to float, categories to stripped text).
            3. Add the ``Age_missing`` and ``Cabin_missing`` indicators, before filling.
            4. Fill missing values with the learned default fill values.
            5. Apply log1p to ``Fare`` (negative fares are first clipped to 0).
            6. Standardize the scaled columns with the learned mean and std.
            7. One-hot encode ``Sex`` and ``Embarked`` with the learned categories.
            8. Return the columns in the fixed ``feature_names`` order, as float32.

        Args:
            df: Any DataFrame with the raw Titanic columns. Extra columns (such as
                ``PassengerId``, ``Name``, ``Ticket`` or ``Survived``) are ignored.

        Returns:
            A float32 DataFrame with the same index as ``df`` and the columns in
            ``feature_names``. It has no missing values.

        Raises:
            NotFittedError: If the preprocessor has not been fitted.
            ValueError: If a required raw column is missing.
        """
        self._check_fitted()
        _check_required_columns(df)
        data = _clean_types(df)

        features = pd.DataFrame(index=df.index)

        # Indicators come first, because filling removes the information they hold.
        missing_flags = {f"{column}_missing": data[column].isna().astype(int) for column in MISSING_INDICATOR_COLUMNS}

        filled = self._apply_log(self._fill_missing(data))

        for column in SCALED_COLUMNS:
            stats = self.scaling[column]
            features[column] = (filled[column] - stats["mean"]) / stats["std"]

        for name, flag in missing_flags.items():
            features[name] = flag

        for column in CATEGORICAL_COLUMNS:
            for category in self.categories[column]:
                features[f"{column}_{category}"] = (filled[column] == category).astype(int)

        return features[self.feature_names].astype(np.float32)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the training set and return its transformed feature table."""
        return self.fit(train_df).transform(train_df)

    def to_dict(self) -> dict:
        """Return the learned values as a plain dict, so they can be saved inside the model file.

        Returns:
            A dict with the format version, fill values, categories, scaling
            statistics and feature names. It contains only strings, numbers,
            lists and dicts.
        """
        self._check_fitted()
        return {
            "format_version": FORMAT_VERSION,
            "fill_values": self.fill_values,
            "categories": self.categories,
            "scaling": self.scaling,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, state: dict) -> TitanicPreprocessor:
        """Rebuild a fitted preprocessor from a dict made by ``to_dict``.

        Args:
            state: The dict returned by ``to_dict``.

        Returns:
            A fitted preprocessor with the saved values.

        Raises:
            ValueError: If the dict was made with a different format version.
        """
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Saved preprocessor has format version {state.get('format_version')}, "
                f"expected {FORMAT_VERSION}. Re-run the training script."
            )
        preprocessor = cls()
        preprocessor.fill_values = state["fill_values"]
        preprocessor.categories = state["categories"]
        preprocessor.scaling = state["scaling"]
        preprocessor.feature_names = state["feature_names"]
        return preprocessor

    def summary(self) -> pd.DataFrame:
        """Return the learned values as a readable table, one row per raw input column."""
        self._check_fitted()
        rows = []
        for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
            stats = self.scaling.get(column, {})
            rows.append(
                {
                    "column": column,
                    "fill value": self.fill_values[column],
                    "log1p": column in LOG_COLUMNS,
                    "scaling mean": stats.get("mean"),
                    "scaling std": stats.get("std"),
                    "categories": self.categories.get(column),
                }
            )
        return pd.DataFrame(rows).set_index("column")

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("The preprocessor is not fitted. Call fit(train_df) or load a saved one first.")

    def _fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in every numeric and categorical input column with the learned values."""
        filled = data.copy()
        for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
            filled[column] = filled[column].fillna(self.fill_values[column])
        return filled

    @staticmethod
    def _apply_log(data: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to the log columns. Negative values are not valid fares, so they are clipped to 0 first."""
        logged = data.copy()
        for column in LOG_COLUMNS:
            logged[column] = np.log1p(logged[column].clip(lower=0))
        return logged

    def _build_feature_names(self) -> list[str]:
        """Output column order: scaled numbers, missing indicators, then one-hot columns."""
        names = list(SCALED_COLUMNS)
        names += [f"{column}_missing" for column in MISSING_INDICATOR_COLUMNS]
        for column in CATEGORICAL_COLUMNS:
            names += [f"{column}_{category}" for category in self.categories[column]]
        return names


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _check_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if any raw column needed by the preprocessor is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Expected the columns of Kaggle's Titanic train.csv.")


def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent types, so data from any CSV is handled the same way.

    Numeric columns become float (text that is not a number becomes missing).
    Text columns (the categorical columns and Cabin) lose surrounding spaces,
    and empty strings become missing.
    """
    data = df[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)
    for column in CATEGORICAL_COLUMNS + ["Cabin"]:
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA).astype(object)
    return data


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw rows into the raw input columns and the target column.

    Args:
        df: Raw Titanic rows that include the ``Survived`` column.

    Returns:
        A tuple (inputs, target): ``inputs`` is ``df`` without ``Survived``;
        ``target`` is ``Survived`` as integers.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' is missing; it is needed to get the target.")
    return df.drop(columns=TARGET), df[TARGET].astype(int)
