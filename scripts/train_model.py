"""
Training script for the Medical Insurance Cost Prediction project.

Reads the preprocessed dataset, splits it into train/test sets, scales numeric
features, trains a Linear Regression baseline, reports error metrics, and saves
the trained model and scaler as artifacts.

Artifacts written:
  - models/linear_regression.joblib   trained LinearRegression model
  - data/scaler.joblib                StandardScaler fitted on training data
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

PROCESSED_FILE = DATA_DIR / "insurance_preprocessed.csv"
SCALER_FILE = DATA_DIR / "scaler.joblib"
MODEL_FILE = MODELS_DIR / "linear_regression.joblib"

# Columns where scale matters for a linear model
NUMERIC_COLS = ["age", "bmi", "children"]
TARGET_COL = "charges"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {path}.\n"
            "Run  python scripts/preprocess_data.py  first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to: {SCALER_FILE}")
    return X_train, X_test, scaler


def train(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained.")
    return model


def evaluate(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Baseline Linear Regression — Test Set Metrics ---")
    print(f"  RMSE : {rmse:>12.2f}")
    print(f"  MAE  : {mae:>12.2f}")
    print(f"  R²   : {r2:>12.4f}")


def save_model(model: LinearRegression) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")


def main() -> None:
    df = load_data(PROCESSED_FILE)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, _ = scale_features(X_train, X_test)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)
    print("\nTraining complete")


if __name__ == "__main__":
    main()
