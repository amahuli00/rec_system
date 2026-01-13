"""
Fallback feature store using cold-start defaults.

This store provides default feature values when the primary store (Redis) is
unavailable or doesn't have data for an entity. It ensures the recommendation
service can continue serving requests even during partial failures.

Design Decisions:

1. **In-memory only**: No external dependencies, always available
   - Can't fail due to network issues
   - Sub-millisecond latency
   - Trade-off: Limited to static default values

2. **Load from feature_metadata.json**: Single source of truth for defaults
   - Same defaults used during training
   - Ensures offline-online consistency
   - No hardcoded values in code

3. **All genres default to 0**: Unknown movies get no genre signal
   - Conservative approach (rather than average genre distribution)
   - Model learns this is a cold-start signal
   - Simple to understand and debug

4. **Demographics defaults**: Neutral values for cold-start users
   - gender=0 (F), age_group=25, occupation=0
   - These are rough population medians
   - Could be improved with population priors
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interface import FeatureStore, FeatureVector, FeatureStoreStats

logger = logging.getLogger(__name__)


class FallbackFeatureStore(FeatureStore):
    """
    In-memory feature store using cold-start default values.

    This store is used when:
    1. Redis is unavailable (circuit breaker open)
    2. An entity doesn't exist in Redis (true cold-start)
    3. As the final fallback in a layered store chain

    It loads default values from feature_metadata.json once at startup
    and returns these defaults for all requests. This ensures the service
    can continue operating even when the primary store fails.

    Example:
        >>> store = FallbackFeatureStore()
        >>> vec = store.get_user_features(999999)  # Unknown user
        >>> print(vec.features["user_avg_rating"])
        3.59  # Global average rating
        >>> print(vec.source)
        "fallback"
    """

    # Default demographic values for cold-start users
    DEFAULT_USER_DEMOGRAPHICS = {
        "gender": 0,        # F (0) is used as default
        "age_group": 25,    # Young adult
        "occupation": 0,    # Other/Unknown
    }

    # All genre features (used for cold-start movies)
    GENRE_FEATURES = [
        "genre_action",
        "genre_adventure",
        "genre_animation",
        "genre_childrens",
        "genre_comedy",
        "genre_crime",
        "genre_documentary",
        "genre_drama",
        "genre_fantasy",
        "genre_film_noir",
        "genre_horror",
        "genre_musical",
        "genre_mystery",
        "genre_romance",
        "genre_sci_fi",
        "genre_thriller",
        "genre_war",
        "genre_western",
    ]

    def __init__(self, metadata_path: Optional[Path] = None):
        """
        Initialize fallback store with cold-start defaults.

        Args:
            metadata_path: Path to feature_metadata.json. If None, uses
                          the default path from ranking/features/.

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is invalid JSON
        """
        if metadata_path is None:
            # Default path relative to this file
            # This file is at: ranking/serving/feature_store/fallback_store.py
            # Metadata is at: ranking/features/feature_metadata.json
            self_path = Path(__file__).resolve()
            metadata_path = (
                self_path.parent.parent.parent / "features" / "feature_metadata.json"
            )

        logger.info(f"Loading cold-start defaults from {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        self.cold_start_defaults = metadata.get("cold_start_defaults", {})

        # Build user cold-start features
        self.default_user_features = {
            "user_rating_count": self.cold_start_defaults.get("user_rating_count", 0.0),
            "user_avg_rating": self.cold_start_defaults.get("user_avg_rating", 3.5),
            "user_rating_std": self.cold_start_defaults.get("user_rating_std", 1.0),
            "user_rating_min": self.cold_start_defaults.get("user_rating_min", 1.0),
            "user_rating_max": self.cold_start_defaults.get("user_rating_max", 5.0),
            **self.DEFAULT_USER_DEMOGRAPHICS,
        }

        # Build movie cold-start features
        self.default_movie_features = {
            "movie_rating_count": self.cold_start_defaults.get("movie_rating_count", 0.0),
            "movie_avg_rating": self.cold_start_defaults.get("movie_avg_rating", 3.5),
            "movie_rating_std": self.cold_start_defaults.get("movie_rating_std", 1.0),
            "movie_rating_min": self.cold_start_defaults.get("movie_rating_min", 1.0),
            "movie_rating_max": self.cold_start_defaults.get("movie_rating_max", 5.0),
        }
        # Add all genres defaulting to 0
        for genre in self.GENRE_FEATURES:
            self.default_movie_features[genre] = 0

        # Stats tracking
        self._stats = FeatureStoreStats()

        logger.info(
            f"FallbackFeatureStore initialized with "
            f"{len(self.default_user_features)} user features, "
            f"{len(self.default_movie_features)} movie features"
        )

    def get_user_features(self, user_id: int) -> Optional[FeatureVector]:
        """
        Get default features for a user.

        This always returns the same default values regardless of user_id,
        since we don't have any information about the user.

        Args:
            user_id: The user ID (ignored, returns defaults)

        Returns:
            FeatureVector with cold-start default values
        """
        start = time.perf_counter()

        # Always return defaults (this is a fallback store)
        features = self.default_user_features.copy()

        latency = (time.perf_counter() - start) * 1000

        # Update stats
        self._stats.total_requests += 1
        self._stats.cache_hits += 1  # "Hits" in the sense that we always have a value

        return FeatureVector(
            features=features,
            source="fallback",
            cache_hit=True,
            latency_ms=latency,
            entity_type="user",
            entity_id=user_id,
        )

    def get_movie_features(self, movie_id: int) -> Optional[FeatureVector]:
        """
        Get default features for a movie.

        This always returns the same default values regardless of movie_id,
        since we don't have any information about the movie.

        Args:
            movie_id: The movie ID (ignored, returns defaults)

        Returns:
            FeatureVector with cold-start default values
        """
        start = time.perf_counter()

        # Always return defaults (this is a fallback store)
        features = self.default_movie_features.copy()

        latency = (time.perf_counter() - start) * 1000

        # Update stats
        self._stats.total_requests += 1
        self._stats.cache_hits += 1

        return FeatureVector(
            features=features,
            source="fallback",
            cache_hit=True,
            latency_ms=latency,
            entity_type="movie",
            entity_id=movie_id,
        )

    def get_batch_user_features(
        self, user_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get default features for multiple users.

        Args:
            user_ids: List of user IDs

        Returns:
            Dictionary mapping each user_id to a FeatureVector
        """
        start = time.perf_counter()

        # Same defaults for all users
        result = {}
        for user_id in user_ids:
            result[user_id] = FeatureVector(
                features=self.default_user_features.copy(),
                source="fallback",
                cache_hit=True,
                latency_ms=0.0,  # Will be updated below
                entity_type="user",
                entity_id=user_id,
            )

        latency = (time.perf_counter() - start) * 1000

        # Update latency for all vectors
        for vec in result.values():
            if vec:
                vec.latency_ms = latency

        # Update stats
        self._stats.total_requests += len(user_ids)
        self._stats.cache_hits += len(user_ids)

        return result

    def get_batch_movie_features(
        self, movie_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get default features for multiple movies.

        Args:
            movie_ids: List of movie IDs

        Returns:
            Dictionary mapping each movie_id to a FeatureVector
        """
        start = time.perf_counter()

        # Same defaults for all movies
        result = {}
        for movie_id in movie_ids:
            result[movie_id] = FeatureVector(
                features=self.default_movie_features.copy(),
                source="fallback",
                cache_hit=True,
                latency_ms=0.0,
                entity_type="movie",
                entity_id=movie_id,
            )

        latency = (time.perf_counter() - start) * 1000

        # Update latency for all vectors
        for vec in result.values():
            if vec:
                vec.latency_ms = latency

        # Update stats
        self._stats.total_requests += len(movie_ids)
        self._stats.cache_hits += len(movie_ids)

        return result

    def health_check(self) -> Dict[str, Any]:
        """
        Check if fallback store is healthy.

        The fallback store is always healthy since it's purely in-memory
        with no external dependencies.

        Returns:
            Health status dictionary
        """
        return {
            "healthy": True,
            "latency_ms": 0.0,
            "message": "Fallback store is always available",
            "type": "fallback",
        }

    def get_stats(self) -> FeatureStoreStats:
        """Get operational statistics."""
        return self._stats
