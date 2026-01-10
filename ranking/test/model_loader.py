"""
Utilities for loading trained models and test data.

This module provides functions to:
- Load XGBoost models from JSON format
- Load feature column names
- Load test data with features

Note:
    Model loading functions are delegated to ranking.shared_utils.
    This module is kept for backwards compatibility and adds test-specific
    functions like load_test_data().
"""

from typing import List, Tuple

import pandas as pd

# Re-export from shared_utils for backwards compatibility
from ranking.shared_utils import (
    MODELS_DIR,
    FEATURES_DIR,
    load_model,
    load_feature_columns,
    load_model_info,
    get_available_models,
)

__all__ = [
    "MODELS_DIR",
    "FEATURES_DIR",
    "load_model",
    "load_feature_columns",
    "load_model_info",
    "load_test_data",
    "get_available_models",
]


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load test features and labels.

    Returns:
        Tuple of:
            - X_test: Feature DataFrame (35 features)
            - df_test: Full DataFrame with user_id, movie_id (needed for NDCG)
            - y_test: Target Series (ratings)
    """
    test_path = FEATURES_DIR / "test_features.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test features not found at {test_path}")

    df_test = pd.read_parquet(test_path)

    # Get feature columns
    feature_cols = load_feature_columns()

    # Extract features and target
    X_test = df_test[feature_cols]
    y_test = df_test["rating"]

    return X_test, df_test, y_test
