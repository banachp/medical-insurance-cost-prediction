"""Reproducibility utilities for model training.

This module centralizes the random seed and split settings used by Mikita's
non-linear modelling scripts.
"""

from __future__ import annotations

import os
import random

import numpy as np

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COLUMN = "charges"


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set global random seeds for deterministic model training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
