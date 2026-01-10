"""
Serving module for the ranking system.

This module provides real-time inference capabilities:
- ServingFeatureBuilder: Builds features for single user + candidate items
- RankerService: End-to-end ranking service
"""

from .ranker_service import RankerService, RankedItem
from .feature_builder import ServingFeatureBuilder

__all__ = [
    "RankerService",
    "RankedItem",
    "ServingFeatureBuilder",
]
