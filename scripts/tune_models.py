"""Tune selected non-linear regression models.

The script tunes the strongest non-linear model families from the comparison
stage using reproducible cross-validation and saves the tuning results.
"""

# AI assistance disclosure:
# Initial code structure and implementation were drafted with ChatGPT
# using GPT-5.5 Thinking. The code was reviewed, edited, and tested
# by Mikita Silivestrau.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import reproducibility
from model_registry import build_model
from sklearn import model_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "insurance_preprocessed.csv"
REPORTS_DIR = REPO_ROOT / "reports"
TUNING_RESULTS_PATH = REPORTS_DIR / "tuning_results.csv"
BEST_PARAMS_PATH = REPORTS_DIR / "best_model_params.json"

N_ITER = 10

PARAMETER_SPACES: dict[str, dict[str, list[Any]]] = {
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 0.9, 1.0],
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", "log2"],
    },
    "ExtraTreesRegressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", "log2"],
    },
    "HistGradientBoostingRegressor": {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 30],
        "l2_regularization": [0.0, 0.01, 0.1],
    },
}


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed insurance dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {DATA_PATH}. "
            "Expected file: data/insurance_preprocessed.csv"
        )

    data = pd.read_csv(DATA_PATH)

    if reproducibility.TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{reproducibility.TARGET_COLUMN}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    y = data[reproducibility.TARGET_COLUMN].astype(float)
    x = data.drop(columns=[reproducibility.TARGET_COLUMN]).copy()

    bool_columns = x.select_dtypes(include=["bool"]).columns
    x[bool_columns] = x[bool_columns].astype(int)

    non_numeric_columns = x.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "All predictors must be numeric after preprocessing. "
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    if x.isna().any().any() or y.isna().any():
        raise ValueError("Missing values found in the preprocessed dataset.")

    return x.astype(float), y


def to_builtin(value: Any) -> Any:
    """Convert NumPy objects into JSON-serializable Python objects."""
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_builtin(item) for item in value]

    return value


def tune_single_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: model_selection.KFold,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune one model and return summarized results."""
    print(f"Tuning {model_name}...")

    model = build_model(
        model_name=model_name,
        seed=reproducibility.RANDOM_SEED,
    )

    search = model_selection.RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAMETER_SPACES[model_name],
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=reproducibility.RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(x_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)

    selected_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "params",
    ]

    results = cv_results[selected_columns].copy()
    results.insert(0, "model", model_name)
    results["mean_test_rmse"] = -results["mean_test_score"]
    results["mean_train_rmse"] = -results["mean_train_score"]
    results["params"] = results["params"].astype(str)

    results = results[
        [
            "model",
            "rank_test_score",
            "mean_test_rmse",
            "std_test_score",
            "mean_train_rmse",
            "params",
        ]
    ].sort_values(["model", "rank_test_score"])

    best_summary = {
        "name": model_name,
        "best_cv_rmse": float(-search.best_score_),
        "best_params": to_builtin(search.best_params_),
    }

    return results, best_summary


def main() -> None:
    """Run reproducible hyperparameter tuning."""
    reproducibility.set_global_seed(reproducibility.RANDOM_SEED)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_preprocessed_data()

    x_train, _, y_train, _ = model_selection.train_test_split(
        x,
        y,
        test_size=reproducibility.TEST_SIZE,
        random_state=reproducibility.RANDOM_SEED,
    )

    cv = model_selection.KFold(
        n_splits=reproducibility.CV_FOLDS,
        shuffle=True,
        random_state=reproducibility.RANDOM_SEED,
    )

    all_results = []
    best_summaries = []

    for model_name in PARAMETER_SPACES:
        model_results, best_summary = tune_single_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            cv=cv,
        )
        all_results.append(model_results)
        best_summaries.append(best_summary)

    tuning_results = pd.concat(all_results, ignore_index=True)
    tuning_results.to_csv(TUNING_RESULTS_PATH, index=False)

    best_model = min(best_summaries, key=lambda item: item["best_cv_rmse"])

    output = {
        "author": "Mikita Silivestrau",
        "random_seed": reproducibility.RANDOM_SEED,
        "test_size": reproducibility.TEST_SIZE,
        "cv_folds": reproducibility.CV_FOLDS,
        "n_iter": N_ITER,
        "scoring": "negative root mean squared error",
        "best_model": best_model,
        "all_model_summaries": best_summaries,
    }

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(output), file, indent=2)

    print("\nBest tuned model:")
    print(json.dumps(to_builtin(best_model), indent=2))
    print(f"\nSaved tuning results to: {TUNING_RESULTS_PATH}")
    print(f"Saved best parameters to: {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
