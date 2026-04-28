"""
Preprocessing script for the Medical Insurance Cost Prediction project.

Reads the raw dataset produced by ingest_data.py, then:
  - Label-encodes binary categoricals: sex, smoker
  - One-hot-encodes the multi-class categorical: region
  - Leaves numeric features (age, bmi, children, charges) untouched

"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_FILE = DATA_DIR / "insurance.csv"
PROCESSED_FILE = DATA_DIR / "insurance_preprocessed.csv"

BINARY_COLS = {
    "sex": {"female": 0, "male": 1},
    "smoker": {"no": 0, "yes": 1},
}


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}.\n"
            "Run  python scripts/ingest_data.py  first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Binary label encoding
    for col, mapping in BINARY_COLS.items():
        df[col] = df[col].str.strip().str.lower().map(mapping)
        if df[col].isna().any():
            raise ValueError(f"Unexpected values in column '{col}'.")

    # One-hot encoding for region
    df = pd.get_dummies(df, columns=["region"], drop_first=False, dtype=int)
    print(f"Categorical encoding done. Columns: {list(df.columns)}")
    return df


def save_outputs(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Preprocessed data saved to: {PROCESSED_FILE}")


def main() -> None:
    df = load_raw(RAW_FILE)
    df = encode_categoricals(df)
    save_outputs(df)
    print("\nPreprocessing complete")


if __name__ == "__main__":
    main()
