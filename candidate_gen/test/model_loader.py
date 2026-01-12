"""
Utilities for loading trained models and test data.

This module provides functions to:
- Load Two-Tower models
- Load FAISS index and embeddings
- Load test data for evaluation

Note:
    Model loading functions are delegated to candidate_gen.shared_utils.
    This module is kept for backwards compatibility and adds test-specific
    functions like load_test_positives().
"""

from typing import Dict, Set

import pandas as pd

# Re-export from shared_utils for backwards compatibility
from candidate_gen.shared_utils import (
    ARTIFACTS_DIR,
    PREPARED_DATA_DIR,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    INDEX_DIR,
    load_model,
    load_model_info,
    load_faiss_index,
    load_user_embeddings,
    load_item_embeddings,
    load_id_mapper,
    get_available_models,
)

__all__ = [
    "ARTIFACTS_DIR",
    "PREPARED_DATA_DIR",
    "MODELS_DIR",
    "load_model",
    "load_model_info",
    "load_faiss_index",
    "load_user_embeddings",
    "load_item_embeddings",
    "load_id_mapper",
    "get_available_models",
    "load_test_positives",
    "load_test_user_positives",
]


def load_test_positives() -> pd.DataFrame:
    """
    Load test set positive interactions.

    Returns:
        DataFrame with user_id, movie_id, rating columns
    """
    test_path = PREPARED_DATA_DIR / "test_positives.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test positives not found at {test_path}")

    return pd.read_parquet(test_path)


def load_test_user_positives() -> Dict[int, Set[int]]:
    """
    Load test set user positive items for evaluation.

    Note: This builds the mapping from test_positives since it's not
    pre-saved during data preparation (only train/val are saved).

    Returns:
        Dict mapping user_idx to set of positive item_idx
    """
    from candidate_gen.data import build_user_positive_items

    test_df = load_test_positives()
    id_mapper = load_id_mapper()

    return build_user_positive_items(test_df, id_mapper)
