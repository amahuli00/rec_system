"""
Ranking service for real-time candidate scoring.

This module provides the RankerService class that orchestrates the full
ranking pipeline: feature building, model inference, and result sorting.
"""

import logging
from dataclasses import dataclass
from typing import List

import xgboost as xgb

from ranking.shared_utils import load_model, load_feature_columns
from .feature_builder import ServingFeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class RankedItem:
    """A ranked item with its score and position."""

    movie_id: int
    predicted_score: float
    rank: int


class RankerService:
    """
    Ranking service for candidate scoring.

    This service provides end-to-end ranking functionality:
    1. Builds features for user + candidate movies
    2. Runs XGBoost model inference
    3. Returns sorted results

    The service is stateful - it loads the model and feature builder once
    at initialization for fast repeated inference.

    Example:
        service = RankerService()
        results = service.rank(user_id=1, candidate_ids=[1, 50, 100, 200])
        for item in results:
            print(f"Rank {item.rank}: Movie {item.movie_id} ({item.predicted_score:.3f})")
    """

    def __init__(self, model_name: str = "xgboost_tuned"):
        """
        Initialize the ranking service.

        Args:
            model_name: Which model to use (default: xgboost_tuned).
                       Available models can be found via get_available_models().

        Loads:
            - XGBoost model
            - Feature column ordering
            - ServingFeatureBuilder with all required data
        """
        logger.info(f"Initializing RankerService with model: {model_name}")

        # Load model
        self.model = load_model(model_name)
        self.model_name = model_name

        # Load feature columns (critical for correct inference)
        self.feature_columns = load_feature_columns()

        # Initialize feature builder
        self.feature_builder = ServingFeatureBuilder()

        logger.info("RankerService initialized successfully")

    def rank(
        self,
        user_id: int,
        candidate_ids: List[int],
    ) -> List[RankedItem]:
        """
        Score and rank candidate items for a user.

        Args:
            user_id: User to rank for
            candidate_ids: List of movie IDs to score

        Returns:
            List of RankedItem sorted by predicted score (descending).
            The best item has rank=1.

        Note:
            - Cold-start users/movies are handled with default features
            - Empty candidate_ids returns empty list
        """
        if not candidate_ids:
            return []

        # Build features
        features = self.feature_builder.build_features(
            user_id=user_id,
            candidate_movie_ids=candidate_ids,
        )

        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(features[self.feature_columns])

        # Get predictions
        predictions = self.model.predict(dmatrix)

        # Create result items
        items = [
            RankedItem(
                movie_id=movie_id,
                predicted_score=float(score),
                rank=0,  # Will be set after sorting
            )
            for movie_id, score in zip(candidate_ids, predictions)
        ]

        # Sort by predicted score (descending)
        items.sort(key=lambda x: x.predicted_score, reverse=True)

        # Assign ranks
        for i, item in enumerate(items):
            item.rank = i + 1

        return items

    def score(
        self,
        user_id: int,
        candidate_ids: List[int],
    ) -> List[float]:
        """
        Get raw scores for candidate items without ranking.

        This is a lightweight alternative to rank() when you only need scores
        and will handle ranking yourself.

        Args:
            user_id: User to score for
            candidate_ids: List of movie IDs to score

        Returns:
            List of predicted scores in the same order as candidate_ids.
        """
        if not candidate_ids:
            return []

        # Build features
        features = self.feature_builder.build_features(
            user_id=user_id,
            candidate_movie_ids=candidate_ids,
        )

        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(features[self.feature_columns])
        predictions = self.model.predict(dmatrix)

        return [float(p) for p in predictions]
