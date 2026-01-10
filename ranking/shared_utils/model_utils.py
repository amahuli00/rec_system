"""
Utilities for loading trained models and related artifacts.

This module provides functions to:
- Load XGBoost models from JSON format
- Load feature column names (critical for inference)
- Load model metadata and feature metadata
"""

import json
from typing import Dict, List

import xgboost as xgb

from .paths import MODELS_DIR, FEATURES_DIR


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
        available = get_available_models()
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

    The order is critical for inference - features must be provided
    in the same order as during training.

    Returns:
        List of feature column names in the correct order
    """
    feature_cols_path = MODELS_DIR / "feature_columns.json"
    with open(feature_cols_path) as f:
        return json.load(f)


def load_model_info(model_name: str = "xgboost_tuned") -> Dict:
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


def load_feature_metadata() -> Dict:
    """
    Load feature metadata including cold-start defaults and feature groups.

    Returns:
        Dictionary with feature metadata
    """
    metadata_path = FEATURES_DIR / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}


def get_available_models() -> List[str]:
    """
    Get list of available model names.

    Returns:
        List of model names (without .json extension)
    """
    return [f.stem for f in MODELS_DIR.glob("xgboost_*.json")]
