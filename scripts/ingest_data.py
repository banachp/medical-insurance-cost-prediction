"""
Data ingestion script for the Medical Insurance Cost Prediction project.

Downloads the 'Medical Cost Personal Datasets' from Kaggle and saves the
raw CSV to the data/ folder.

Requirements:
    - kaggle Python package installed  (`pip install kaggle`)
    - Kaggle API credentials configured at ~/.kaggle/kaggle.json
      (https://www.kaggle.com/settings → API → Create New Token)

Usage:
    python scripts/ingest_data.py
"""

import os
import zipfile
from pathlib import Path

# Pointing the Kaggle client at the credentials file stored in this repo
_KAGGLE_DIR = Path(__file__).resolve().parents[1] / ".kaggle"
os.environ.setdefault("KAGGLE_CONFIG_DIR", str(_KAGGLE_DIR))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET = "mirichoi0218/insurance"
RAW_FILE = "insurance.csv"


def download_dataset(dest_dir: Path) -> None:
    try:
        import kaggle
    except ImportError:
        raise SystemExit(
            "The 'kaggle' package is not installed. " "Run:  pip install kaggle"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{DATASET}' from Kaggle …")
    # kaggle.api.dataset_download_files downloads a zip to dest_dir
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET,
        path=str(dest_dir),
        unzip=False,
        quiet=False,
    )

    # The downloaded archive is named after the dataset slug
    zip_name = DATASET.split("/")[-1] + ".zip"
    zip_path = dest_dir / zip_name

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected archive not found at {zip_path}. "
            "Check your Kaggle credentials and dataset name."
        )

    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    zip_path.unlink()  # removing the zip, keeping only the CSV
    print(f"Removed archive {zip_path.name}")


def verify(dest_dir: Path) -> None:
    csv_path = dest_dir / RAW_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file '{RAW_FILE}' not found in {dest_dir}.")

    import csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    expected_columns = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = expected_columns - {col.strip().lower() for col in header}
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    print(f"Verification passed – {row_count} rows, columns: {header}")


def main() -> None:
    download_dataset(DATA_DIR)
    verify(DATA_DIR)
    print(f"\nData saved to: {DATA_DIR / RAW_FILE}")


if __name__ == "__main__":
    main()
