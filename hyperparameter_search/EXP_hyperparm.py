"""Select the MLP's hyperparameters with 5-fold cross-validation on the training set.

What it does:
    1. Loads the full dataset and holds out the test set with the shared split
       (the test set is not used here at all).
    2. Builds 5 stratified folds on the training set once, and reuses them everywhere.
    3. Trains every configuration of an 18-point grid (depth, width, dropout)
       on every fold through ``fit_and_train`` from ``train.py``,
       so the search runs exactly the code that ``train.py`` ships. Each run is
       scored by ROC AUC on the fold's held-out part.
    4. Scores an untuned tree-based baseline (HistGradientBoostingClassifier with
       minimal preprocessing) on the same folds, for reference.
    5. Selects the configuration with the highest mean AUC (see ``select_configuration``),
       writes the results to CSV and prints a summary with the constants to set.

It does not change ``train.py`` or ``src/``: after reading the summary, update the
constants by hand and re-run ``python train.py``.

Usage (from the project root):
    python hyperparameter_search/EXP_hyperparm.py
    python hyperparameter_search/EXP_hyperparm.py --data data/train.csv --output-dir hyperparameter_search/results/

Writes (overwritten on each run):
    hyperparameter_search/results/mlp_cv_results.csv       one row per MLP configuration
    hyperparameter_search/results/baseline_cv_results.csv  one row for the baseline
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# This script lives in hyperparameter_search/; add the project root to sys.path so that
# train.py and src/ can be imported, whichever folder the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_split import split_dataset  # noqa: E402
from src.model import predict_proba  # noqa: E402
from train import DEFAULT_DATA_PATH, SEED, TARGET, TEST_SIZE, fit_and_train, load_data  # noqa: E402

N_FOLDS = 5

# The search grid: 2 x 3 x 3 = 18 configurations.
DEPTHS = [1, 2]
WIDTHS = [16, 32, 64]
DROPOUTS = [0.0, 0.2, 0.4]

# Columns kept for the baseline, and the ones treated as categories.
BASELINE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
BASELINE_CATEGORICAL = ["Sex", "Embarked"]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FOLD_AUC_COLUMNS = [f"auc_fold_{fold}" for fold in range(1, N_FOLDS + 1)]

logger = logging.getLogger("EXP_hyperparm")

Folds = list[tuple[np.ndarray, np.ndarray]]


def build_folds(train_df: pd.DataFrame) -> Folds:
    """Create the stratified folds once.

    Args:
        train_df: The training set.

    Returns:
        A list of ``(train_index, validation_index)`` pairs of row positions in ``train_df``.
    """
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(train_df, train_df[TARGET]))


def build_grid() -> list[dict]:
    """Return every configuration of the grid as a dict, in a fixed order."""
    return [
        {"depth": depth, "width": width, "dropout": dropout}
        for depth, width, dropout in itertools.product(DEPTHS, WIDTHS, DROPOUTS)
    ]


def standard_error(values: list[float]) -> float:
    """Standard error of the mean: std (ddof=1) divided by the square root of the count."""
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def run_mlp_configuration(train_df: pd.DataFrame, folds: Folds, config: dict) -> dict:
    """Train one configuration on every fold and summarise its results.

    Every run goes through ``fit_and_train``; settings not in ``config`` use the
    defaults from ``train.py``.

    Args:
        train_df: The training set.
        folds: The shared fold list.
        config: ``depth``, ``width`` and ``dropout``.

    Returns:
        One result row: the configuration, ``n_parameters``, the AUC per fold,
        ``auc_mean``, ``auc_se`` and the best-epoch mean, min and max.
    """
    aucs, best_epochs = [], []
    n_parameters = 0
    for train_index, validation_index in folds:
        fold_train_set = train_df.iloc[train_index]
        fold_validation_set = train_df.iloc[validation_index]
        model, preprocessor, history = fit_and_train(
            fold_train_set,
            fold_validation_set,
            hidden_sizes=[config["width"]] * config["depth"],
            dropout=config["dropout"],
        )
        # Scored with the early-stopping best weights returned by fit_and_train.
        probabilities = predict_proba(model, preprocessor, fold_validation_set)
        aucs.append(roc_auc_score(fold_validation_set[TARGET], probabilities))
        best_epochs.append(history["best_epoch"])
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        **config,
        "n_parameters": n_parameters,
        **dict(zip(FOLD_AUC_COLUMNS, aucs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_se": standard_error(aucs),
        "best_epoch_mean": float(np.mean(best_epochs)),
        "best_epoch_min": int(min(best_epochs)),
        "best_epoch_max": int(max(best_epochs)),
    }


def run_mlp_search(train_df: pd.DataFrame, folds: Folds) -> pd.DataFrame:
    """Run every grid configuration over all folds, logging one line per configuration.

    Returns:
        One row per configuration, in grid order.
    """
    grid = build_grid()
    rows = []
    for number, config in enumerate(grid, start=1):
        row = run_mlp_configuration(train_df, folds, config)
        rows.append(row)
        logger.info(
            "Config %d/%d depth=%d width=%d dropout=%s: AUC %.3f +/- %.3f (SE)",
            number, len(grid), config["depth"], config["width"], config["dropout"],
            row["auc_mean"], row["auc_se"],
        )
    return pd.DataFrame(rows)


def baseline_features(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Minimal preprocessing for the baseline.

    Keeps the baseline columns and turns ``Sex`` and ``Embarked`` into pandas
    categories with a fixed category list; values outside that list become missing.
    Nothing is filled, scaled or added.

    Args:
        df: Raw Titanic rows.
        categories: Category list per categorical column, taken from ``fold_train_set``.
    """
    features = df[BASELINE_COLUMNS].copy()
    for column in BASELINE_CATEGORICAL:
        features[column] = pd.Categorical(features[column], categories=categories[column])
    return features


def run_baseline(train_df: pd.DataFrame, folds: Folds) -> pd.DataFrame:
    """Score the untuned HistGradientBoostingClassifier on the shared folds.

    Returns:
        One row: ``model``, the AUC per fold, ``auc_mean`` and ``auc_se``.
    """
    aucs = []
    for train_index, validation_index in folds:
        fold_train_set = train_df.iloc[train_index]
        fold_validation_set = train_df.iloc[validation_index]
        # Category lists come from fold_train_set only.
        categories = {column: sorted(fold_train_set[column].dropna().unique()) for column in BASELINE_CATEGORICAL}
        model = HistGradientBoostingClassifier(random_state=SEED, categorical_features="from_dtype")
        model.fit(baseline_features(fold_train_set, categories), fold_train_set[TARGET])
        probabilities = model.predict_proba(baseline_features(fold_validation_set, categories))[:, 1]
        aucs.append(roc_auc_score(fold_validation_set[TARGET], probabilities))

    row = {
        "model": "HistGradientBoostingClassifier (defaults)",
        **dict(zip(FOLD_AUC_COLUMNS, aucs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_se": standard_error(aucs),
    }
    return pd.DataFrame([row])


def select_configuration(results: pd.DataFrame) -> pd.DataFrame:
    """Mark the selected configuration: the one with the highest ``auc_mean``.

    If several configurations share the highest ``auc_mean``, the one with the
    fewest trainable parameters is selected.

    Args:
        results: One row per configuration.

    Returns:
        A copy with an ``is_selected`` column, sorted by ``auc_mean`` descending.
    """
    results = results.copy()
    best = results[results["auc_mean"] == results["auc_mean"].max()]
    selected_index = best["n_parameters"].idxmin()
    results["is_selected"] = results.index == selected_index
    return results.sort_values("auc_mean", ascending=False, kind="stable").reset_index(drop=True)


def write_outputs(mlp_results: pd.DataFrame, baseline_results: pd.DataFrame, output_dir: Path) -> None:
    """Write both result tables as CSV files to ``output_dir`` (created if missing)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mlp_columns = (
        ["depth", "width", "dropout", "n_parameters"]
        + FOLD_AUC_COLUMNS
        + ["auc_mean", "auc_se", "best_epoch_mean", "best_epoch_min", "best_epoch_max", "is_selected"]
    )
    mlp_results[mlp_columns].to_csv(output_dir / "mlp_cv_results.csv", index=False)
    baseline_results[["model"] + FOLD_AUC_COLUMNS + ["auc_mean", "auc_se"]].to_csv(
        output_dir / "baseline_cv_results.csv", index=False
    )
    logger.info("Wrote %s", output_dir / "mlp_cv_results.csv")
    logger.info("Wrote %s", output_dir / "baseline_cv_results.csv")


def print_summary(mlp_results: pd.DataFrame, baseline_results: pd.DataFrame) -> None:
    """Print the top configurations, the selection, the baseline and the constants to set."""
    selected = mlp_results[mlp_results["is_selected"]].iloc[0]
    baseline = baseline_results.iloc[0]
    table_columns = ["depth", "width", "dropout", "n_parameters", "auc_mean", "auc_se",
                     "best_epoch_min", "best_epoch_max", "is_selected"]

    print("\n=== Top 10 configurations by auc_mean ===")
    print(mlp_results[table_columns].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Selected configuration (highest auc_mean) ===")
    print(f"depth={selected['depth']} width={selected['width']} dropout={selected['dropout']}")
    print(
        f"auc_mean={selected['auc_mean']:.4f} auc_se={selected['auc_se']:.4f} "
        f"n_parameters={selected['n_parameters']} best_epoch range={selected['best_epoch_min']}-{selected['best_epoch_max']}"
    )

    print("\n=== Baseline ===")
    print(f"{baseline['model']}: auc_mean={baseline['auc_mean']:.4f} auc_se={baseline['auc_se']:.4f}")

    print("\n=== Set by hand ===")
    print(f"src/model.py: HIDDEN_SIZES = {[int(selected['width'])] * int(selected['depth'])}")
    print(f"src/model.py: DROPOUT = {selected['dropout']}")


def main() -> None:
    """Run the search: data, folds, MLP grid, baseline, selection, outputs, summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Select the MLP's hyperparameters with 5-fold cross-validation.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / DEFAULT_DATA_PATH, help="Path to Kaggle's train.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for the result CSV files.")
    args = parser.parse_args()
    start_time = time.perf_counter()

    # 1. Load the data and hold out the test set.
    full_df = load_data(args.data)
    train_df, test_df = split_dataset(full_df, TEST_SIZE, SEED)
    # test_df is not used anywhere after this line.
    logger.info("train_df=%d rows (the test set is held out and not used)", len(train_df))

    # 2. Build the folds once; every model below uses this same list.
    folds = build_folds(train_df)
    logger.info("Built %d stratified folds", len(folds))

    # 3. MLP grid search.
    mlp_results = select_configuration(run_mlp_search(train_df, folds))

    # 4. Baseline on the same folds.
    baseline_results = run_baseline(train_df, folds)
    logger.info("Baseline: AUC %.3f +/- %.3f (SE)", baseline_results.loc[0, "auc_mean"], baseline_results.loc[0, "auc_se"])

    # 5. Outputs and summary.
    write_outputs(mlp_results, baseline_results, args.output_dir)
    print_summary(mlp_results, baseline_results)
    logger.info("Total run time: %.1f minutes", (time.perf_counter() - start_time) / 60)


if __name__ == "__main__":
    main()
