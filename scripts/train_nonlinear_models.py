"""Train and compare non-linear regression models.

The script uses the preprocessed insurance dataset created in the previous
pipeline step, trains several non-linear regressors, evaluates them on the same
deterministic train/test split, and saves the comparison table.
"""

# AI assistance disclosure:
# Initial structure and implementation were drafted with ChatGPT
# using GPT-5.5 Thinking. The code was reviewed,
# edited, and tested by Mikita Silivestrau.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import reproducibility
from model_registry import get_default_models
from sklearn import metrics, model_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "insurance_preprocessed.csv"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "model_comparison.csv"


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


def evaluate_model(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | str]:
    """Train one model and calculate test-set metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model_name,
        "mae": float(metrics.mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_test, predictions))),
        "r2": float(metrics.r2_score(y_test, predictions)),
    }


def main() -> None:
    """Run model comparison and save the results."""
    reproducibility.set_global_seed(reproducibility.RANDOM_SEED)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_preprocessed_data()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=reproducibility.TEST_SIZE,
        random_state=reproducibility.RANDOM_SEED,
    )

    results = []
    models = get_default_models(seed=reproducibility.RANDOM_SEED)

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        result = evaluate_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
