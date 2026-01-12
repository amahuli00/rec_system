"""
Utilities for loading trained Two-Tower models and related artifacts.

This module provides functions to:
- Load Two-Tower PyTorch models
- Load FAISS index for retrieval
- Load user/item embeddings
- Load ID mapper for ID conversion
- Get available models
"""

import json
from typing import Dict, List

import faiss
import numpy as np
import torch

from ..data import IDMapper
from ..model import ModelConfig, TwoTowerModel
from .paths import MODELS_DIR, EMBEDDINGS_DIR, INDEX_DIR, PREPARED_DATA_DIR


def load_model(model_name: str = "two_tower_model", device: str = "cpu") -> TwoTowerModel:
    """
    Load a Two-Tower model from saved state dict.

    Args:
        model_name: Name of the model file (without .pt extension)
        device: Device to load model onto

    Returns:
        Loaded TwoTowerModel instance

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        available = get_available_models()
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. "
            f"Available models: {available}"
        )

    # Load model info for architecture params
    model_info = load_model_info()

    # Create model with correct architecture
    config = ModelConfig(
        embedding_dim=model_info["embedding_dim"],
        use_mlp=model_info.get("use_mlp", False),
    )
    model = TwoTowerModel(
        num_users=model_info["num_users"],
        num_items=model_info["num_items"],
        config=config,
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def load_model_info() -> Dict:
    """
    Load model metadata/info.

    Returns:
        Dictionary with model metadata (num_users, num_items, embedding_dim, etc.)
    """
    info_path = MODELS_DIR / "model_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Model info not found at {info_path}")

    with open(info_path) as f:
        return json.load(f)


def load_faiss_index() -> faiss.Index:
    """
    Load FAISS index from disk.

    Returns:
        FAISS index for item embeddings

    Raises:
        FileNotFoundError: If index file doesn't exist
    """
    index_path = INDEX_DIR / "item_index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    return faiss.read_index(str(index_path))


def load_index_metadata() -> Dict:
    """
    Load FAISS index metadata.

    Returns:
        Dictionary with index metadata (index_type, metric, num_vectors, etc.)
    """
    metadata_path = INDEX_DIR / "index_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}


def load_user_embeddings() -> np.ndarray:
    """
    Load pre-computed user embeddings.

    Returns:
        Numpy array of shape [num_users, embedding_dim]
    """
    embeddings_path = EMBEDDINGS_DIR / "user_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"User embeddings not found at {embeddings_path}")

    return np.load(embeddings_path)


def load_item_embeddings() -> np.ndarray:
    """
    Load pre-computed item embeddings.

    Returns:
        Numpy array of shape [num_items, embedding_dim]
    """
    embeddings_path = EMBEDDINGS_DIR / "item_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Item embeddings not found at {embeddings_path}")

    return np.load(embeddings_path)


def load_embedding_metadata() -> Dict:
    """
    Load embedding metadata.

    Returns:
        Dictionary with embedding metadata (num_users, num_items, embedding_dim, etc.)
    """
    metadata_path = EMBEDDINGS_DIR / "embedding_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}


def load_id_mapper() -> IDMapper:
    """
    Load ID mapper for user/item ID conversion.

    Returns:
        IDMapper instance
    """
    return IDMapper.load(PREPARED_DATA_DIR)


def get_available_models() -> List[str]:
    """
    Get list of available model names.

    Returns:
        List of model names (without .pt extension)
    """
    return [f.stem for f in MODELS_DIR.glob("*.pt") if not f.stem.startswith("checkpoint")]
