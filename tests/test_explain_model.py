"""Tests for model explainability helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import explain_model  # noqa: E402


def test_validate_feature_columns_accepts_matching_columns() -> None:
    """Feature validation should accept the exact metadata order."""
    x = pd.DataFrame(columns=["age", "sex", "bmi"])
    metadata = {"feature_columns": ["age", "sex", "bmi"]}

    explain_model.validate_feature_columns(x, metadata)


def test_validate_feature_columns_requires_metadata() -> None:
    """Missing feature metadata should be reported clearly."""
    x = pd.DataFrame(columns=["age"])

    with pytest.raises(ValueError, match="feature_columns"):
        explain_model.validate_feature_columns(x, {})
