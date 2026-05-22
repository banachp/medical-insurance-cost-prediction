"""Tests for the final model evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_model  # noqa: E402


def test_calculate_metrics_returns_expected_regression_values() -> None:
    """Regression metrics should match simple hand-checkable inputs."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    predictions = np.array([12.0, 18.0, 33.0])

    metric_values = evaluate_model.calculate_metrics(y_true, predictions)

    assert metric_values["mae"] == pytest.approx(7.0 / 3.0)
    assert metric_values["rmse"] == pytest.approx(np.sqrt(17.0 / 3.0))
    assert metric_values["r2"] == pytest.approx(0.915)


def test_validate_feature_columns_rejects_mismatched_order() -> None:
    """Feature validation should fail when metadata order is not respected."""
    x = pd.DataFrame(columns=["age", "bmi"])
    metadata = {"feature_columns": ["bmi", "age"]}

    with pytest.raises(ValueError, match="Feature columns do not match"):
        evaluate_model.validate_feature_columns(x, metadata)
