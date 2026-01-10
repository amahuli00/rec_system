"""
Model module for Two-Tower candidate generation.

This module provides:
- Tower: Simple embedding tower with L2 normalization
- TwoTowerModel: Complete model with user and item towers
"""

from .config import ModelConfig
from .tower import Tower
from .two_tower import TwoTowerModel

__all__ = [
    "ModelConfig",
    "Tower",
    "TwoTowerModel",
]
