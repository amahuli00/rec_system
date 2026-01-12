"""
Shared utilities for the candidate generation system.

This module provides common utilities used across candidate generation components:
- Path constants for project directories
- Model and artifact loading utilities
"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    CANDIDATE_GEN_DIR,
    ARTIFACTS_DIR,
    PREPARED_DATA_DIR,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    INDEX_DIR,
)
from .model_utils import (
    load_model,
    load_model_info,
    load_faiss_index,
    load_index_metadata,
    load_user_embeddings,
    load_item_embeddings,
    load_embedding_metadata,
    load_id_mapper,
    get_available_models,
)

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "CANDIDATE_GEN_DIR",
    "ARTIFACTS_DIR",
    "PREPARED_DATA_DIR",
    "MODELS_DIR",
    "EMBEDDINGS_DIR",
    "INDEX_DIR",
    # Model utilities
    "load_model",
    "load_model_info",
    "load_faiss_index",
    "load_index_metadata",
    "load_user_embeddings",
    "load_item_embeddings",
    "load_embedding_metadata",
    "load_id_mapper",
    "get_available_models",
]
