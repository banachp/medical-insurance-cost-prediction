"""Train, evaluate, and save the final tuned model.

The script loads the best hyperparameters from the tuning stage, trains the
selected model on the deterministic training split, evaluates it on the held-out
test split, and saves the trained model artifact with metadata.
"""

# AI assistance disclosure:
# Initial code structure and implementation were drafted with ChatGPT
# using GPT-5.5 Thinking. The code was reviewed, edited, and tested
# by Mikita Silivestrau.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import reproducibility
from model_registry import build_model
from sklearn import metrics, model_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "insurance_preprocessed.csv"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

BEST_PARAMS_PATH = REPORTS_DIR / "best_model_params.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_METADATA_PATH = MODELS_DIR / "final_model_metadata.json"
FINAL_METRICS_PATH = REPORTS_DIR / "final_model_test_metrics.csv"


def file_sha256(path: Path) -> str:
    """Calculate a SHA-256 hash for a file."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


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


def load_best_model_config() -> dict[str, Any]:
    """Load the selected model name and best hyperparameters."""
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Best model parameters not found at {BEST_PARAMS_PATH}. "
            "Run scripts/tune_models.py first."
        )

    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for the held-out test set."""
    return {
        "mae": float(metrics.mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, predictions))),
        "r2": float(metrics.r2_score(y_true, predictions)),
    }


def main() -> None:
    """Train, evaluate, and save the final tuned model."""
    reproducibility.set_global_seed(reproducibility.RANDOM_SEED)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_preprocessed_data()
    config = load_best_model_config()

    best_model = config["best_model"]
    model_name = best_model["name"]
    model_params = best_model["best_params"]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=reproducibility.TEST_SIZE,
        random_state=reproducibility.RANDOM_SEED,
    )

    model = build_model(
        model_name=model_name,
        params=model_params,
        seed=reproducibility.RANDOM_SEED,
    )

    print(f"Training final model: {model_name}")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, predictions)

    joblib.dump(model, FINAL_MODEL_PATH)

    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
            }
        ]
    )
    metrics_df.to_csv(FINAL_METRICS_PATH, index=False)

    metadata = {
        "author": "Mikita Silivestrau",
        "random_seed": reproducibility.RANDOM_SEED,
        "test_size": reproducibility.TEST_SIZE,
        "target_column": reproducibility.TARGET_COLUMN,
        "model_name": model_name,
        "model_params": model_params,
        "metrics_on_test_split": test_metrics,
        "n_rows_total": int(len(x)),
        "n_rows_train": int(len(x_train)),
        "n_rows_test": int(len(x_test)),
        "feature_columns": list(x.columns),
        "data_path": str(DATA_PATH.relative_to(REPO_ROOT)),
        "data_sha256": file_sha256(DATA_PATH),
        "best_params_path": str(BEST_PARAMS_PATH.relative_to(REPO_ROOT)),
        "final_model_path": str(FINAL_MODEL_PATH.relative_to(REPO_ROOT)),
    }

    with FINAL_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nFinal model test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved final model to: {FINAL_MODEL_PATH}")
    print(f"Saved final model metadata to: {FINAL_METADATA_PATH}")
    print(f"Saved final model metrics to: {FINAL_METRICS_PATH}")


if __name__ == "__main__":
    main()
