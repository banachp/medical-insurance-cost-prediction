"""Evaluate the saved final model on the deterministic test split."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import reproducibility
from sklearn import metrics, model_selection

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "medical_insurance_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "medical_insurance_cache"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "insurance_preprocessed.csv"
MODEL_PATH = REPO_ROOT / "models" / "final_model.joblib"
METADATA_PATH = REPO_ROOT / "models" / "final_model_metadata.json"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_PATH = REPORTS_DIR / "final_evaluation_metrics.csv"
PREDICTIONS_PATH = REPORTS_DIR / "final_evaluation_predictions.csv"
EVALUATION_FIGURE_PATH = FIGURES_DIR / "final_model_evaluation.png"


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load metadata saved with the final model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Final model metadata not found at {path}. "
            "Run scripts/save_final_model.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_preprocessed_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed dataset and separate predictors from target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {path}. "
            "Run scripts/preprocess_data.py first."
        )

    data = pd.read_csv(path)
    target_column = reproducibility.TARGET_COLUMN

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    y = data[target_column].astype(float)
    x = data.drop(columns=[target_column]).copy()

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


def validate_feature_columns(
    x: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Confirm the evaluation frame matches the model training metadata."""
    expected_columns = metadata.get("feature_columns")
    if not expected_columns:
        raise ValueError("Model metadata does not define feature_columns.")

    actual_columns = list(x.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            "Feature columns do not match final model metadata. "
            f"Expected: {expected_columns}. Actual: {actual_columns}."
        )


def split_test_data(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the deterministic held-out test split."""
    _, x_test, _, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=reproducibility.TEST_SIZE,
        random_state=reproducibility.RANDOM_SEED,
    )
    return x_test, y_test


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load the final model artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Final model not found at {path}. Run scripts/save_final_model.py first."
        )

    return joblib.load(path)


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for final evaluation."""
    return {
        "mae": float(metrics.mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, predictions))),
        "r2": float(metrics.r2_score(y_true, predictions)),
    }


def save_evaluation_outputs(
    metadata: dict[str, Any],
    y_test: pd.Series,
    predictions: np.ndarray,
    metric_values: dict[str, float],
) -> None:
    """Persist final metrics, row-level predictions, and evaluation charts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {
                "model": metadata["model_name"],
                "mae": metric_values["mae"],
                "rmse": metric_values["rmse"],
                "r2": metric_values["r2"],
                "n_rows_test": int(len(y_test)),
                "random_seed": reproducibility.RANDOM_SEED,
                "test_size": reproducibility.TEST_SIZE,
            }
        ]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    predictions_df = pd.DataFrame(
        {
            "actual_charges": y_test.to_numpy(),
            "predicted_charges": predictions,
            "residual": y_test.to_numpy() - predictions,
        },
        index=y_test.index,
    ).sort_index()
    predictions_df.to_csv(PREDICTIONS_PATH, index_label="row_id")

    plot_evaluation(predictions_df)


def plot_evaluation(predictions_df: pd.DataFrame) -> None:
    """Create actual-vs-predicted and residual charts."""
    actual = predictions_df["actual_charges"]
    predicted = predictions_df["predicted_charges"]
    residual = predictions_df["residual"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(actual, predicted, alpha=0.7, edgecolor="none")
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    axes[0].plot([min_value, max_value], [min_value, max_value], color="black")
    axes[0].set_title("Actual vs predicted charges")
    axes[0].set_xlabel("Actual charges")
    axes[0].set_ylabel("Predicted charges")

    axes[1].scatter(predicted, residual, alpha=0.7, edgecolor="none")
    axes[1].axhline(0, color="black")
    axes[1].set_title("Residuals by predicted charges")
    axes[1].set_xlabel("Predicted charges")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig(EVALUATION_FIGURE_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run final model evaluation."""
    metadata = load_metadata()
    x, y = load_preprocessed_data()
    validate_feature_columns(x, metadata)

    x_test, y_test = split_test_data(x, y)
    model = load_model()
    predictions = model.predict(x_test)
    metric_values = calculate_metrics(y_test, predictions)

    save_evaluation_outputs(
        metadata=metadata,
        y_test=y_test,
        predictions=predictions,
        metric_values=metric_values,
    )

    print("Final model evaluation metrics:")
    print(json.dumps(metric_values, indent=2))
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")
    print(f"Saved evaluation chart to: {EVALUATION_FIGURE_PATH}")


if __name__ == "__main__":
    main()
