"""
Utilities for loading trained models and test data.

This module provides functions to:
- Load XGBoost models from JSON format
- Load feature column names
- Load test data with features
"""

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import xgboost as xgb

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "ranking" / "models"
FEATURES_DIR = PROJECT_ROOT / "ranking" / "features"


def load_model(model_name: str) -> xgb.Booster:
    """
    Load an XGBoost model from JSON format.

    Args:
        model_name: Name of the model (e.g., "xgboost_tuned", "xgboost_baseline")

    Returns:
        Loaded XGBoost Booster model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = MODELS_DIR / f"{model_name}.json"
    if not model_path.exists():
        available = [f.stem for f in MODELS_DIR.glob("xgboost_*.json")]
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. "
            f"Available models: {available}"
        )

    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def load_feature_columns() -> List[str]:
    """
    Load the ordered list of feature column names.

    Returns:
        List of feature column names in the correct order
    """
    feature_cols_path = MODELS_DIR / "feature_columns.json"
    with open(feature_cols_path) as f:
        return json.load(f)


def load_model_info(model_name: str = "xgboost_tuned") -> dict:
    """
    Load model metadata/info.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model metadata (params, metrics, etc.)
    """
    info_path = MODELS_DIR / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}


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


def get_available_models() -> List[str]:
    """
    Get list of available model names.

    Returns:
        List of model names (without .json extension)
    """
    return [f.stem for f in MODELS_DIR.glob("xgboost_*.json")]
