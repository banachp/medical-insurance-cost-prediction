"""Model registry for non-linear regression models.

The project baseline is a linear regression model. This registry defines
non-linear scikit-learn models used for comparison and tuning.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from reproducibility import RANDOM_SEED


def get_default_models(seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Return fresh non-linear model instances with fixed random states."""
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=seed,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=300,
        ),
    }


def build_model(
    model_name: str,
    params: dict[str, Any] | None = None,
    seed: int = RANDOM_SEED,
) -> Any:
    """Build a model by name and optionally apply hyperparameters."""
    models = get_default_models(seed=seed)

    if model_name not in models:
        available = ", ".join(models)
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    model = models[model_name]

    if params:
        model.set_params(**params)

    return model