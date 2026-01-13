"""
Unified recommendation serving module.

This module provides a high-level API for serving recommendations by orchestrating
the candidate generation and ranking stages of the recommendation pipeline.
"""

from serving.recommendation_service import RecommendationService, RecommendedItem

__all__ = ['RecommendationService', 'RecommendedItem']
