"""Generate model explainability outputs for the final trained model."""

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
from sklearn import inspection, model_selection

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

FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
PERMUTATION_FIGURE_PATH = FIGURES_DIR / "permutation_importance.png"
PARTIAL_DEPENDENCE_FIGURE_PATH = FIGURES_DIR / "partial_dependence_top_features.png"

N_REPEATS = 20
TOP_FEATURE_COUNT = 5
PARTIAL_DEPENDENCE_FEATURE_COUNT = 3


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
    """Confirm the explanation frame matches the model training metadata."""
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


def calculate_permutation_importance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Calculate permutation importance on the deterministic test split."""
    result = inspection.permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=N_REPEATS,
        random_state=reproducibility.RANDOM_SEED,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    return importance_df[["rank", "feature", "importance_mean", "importance_std"]]


def plot_permutation_importance(importance_df: pd.DataFrame) -> None:
    """Plot the top permutation importance values."""
    top_features = importance_df.head(TOP_FEATURE_COUNT).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
    )
    ax.set_title("Top permutation importance scores")
    ax.set_xlabel("Increase in RMSE after permutation")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(PERMUTATION_FIGURE_PATH, dpi=160)
    plt.close(fig)


def plot_partial_dependence(
    model: Any,
    x_test: pd.DataFrame,
    top_feature_names: list[str],
) -> None:
    """Plot partial dependence for the most important features."""
    if not top_feature_names:
        raise ValueError("At least one feature is required for partial dependence.")

    display = inspection.PartialDependenceDisplay.from_estimator(
        model,
        x_test,
        features=top_feature_names,
        kind="average",
        grid_resolution=30,
    )
    display.figure_.set_size_inches(12, 4)
    display.figure_.suptitle("Partial dependence for top features")
    display.figure_.tight_layout()
    display.figure_.savefig(PARTIAL_DEPENDENCE_FIGURE_PATH, dpi=160)
    plt.close(display.figure_)


def save_explainability_outputs(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Persist feature importance and explainability charts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = calculate_permutation_importance(model, x_test, y_test)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    plot_permutation_importance(importance_df)
    top_partial_features = importance_df.head(PARTIAL_DEPENDENCE_FEATURE_COUNT)[
        "feature"
    ].tolist()
    plot_partial_dependence(model, x_test, top_partial_features)

    return importance_df


def main() -> None:
    """Run model explainability generation."""
    metadata = load_metadata()
    x, y = load_preprocessed_data()
    validate_feature_columns(x, metadata)

    x_test, y_test = split_test_data(x, y)
    model = load_model()
    importance_df = save_explainability_outputs(model, x_test, y_test)

    print("Top feature importances:")
    print(importance_df.head(TOP_FEATURE_COUNT).to_string(index=False))
    print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved permutation importance chart to: {PERMUTATION_FIGURE_PATH}")
    print(f"Saved partial dependence chart to: {PARTIAL_DEPENDENCE_FIGURE_PATH}")


if __name__ == "__main__":
    main()
