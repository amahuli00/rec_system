"""
Data preparation module for Two-Tower candidate generation.

This module handles:
- ID mappings (user_id, movie_id → contiguous indices)
- Positive interaction filtering (rating >= threshold)
- PyTorch Dataset for training
"""

from .config import DataConfig
from .dataset import IDMapper, TwoTowerDataset, build_user_positive_items

__all__ = [
    "DataConfig",
    "IDMapper",
    "TwoTowerDataset",
    "build_user_positive_items",
]
