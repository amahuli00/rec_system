"""
Feature store interface definition.

This module defines the abstract interface for feature storage backends.
All feature store implementations (Redis, in-memory, etc.) must implement
this interface to ensure consistent behavior across the system.

Design Principles:
1. Fail gracefully: Return None on errors rather than raising exceptions
   - This enables fallback behavior without try/except everywhere
   - Callers can decide how to handle missing features

2. Observability built-in: Return metadata (source, latency, cache_hit)
   - Enables monitoring and debugging without additional instrumentation
   - Helps identify performance bottlenecks

3. Batch operations: Support fetching multiple entities in one call
   - Reduces network round-trips (critical for Redis latency)
   - Enables pipeline optimizations in implementations

4. Explicit feature groups: User stats, demographics, movie stats, genres
   - Mirrors the feature engineering structure
   - Allows selective retrieval for different use cases
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """
    Container for feature values with metadata.

    This dataclass wraps feature values along with observability metadata
    that helps monitor and debug the feature serving system.

    Attributes:
        features: Dictionary mapping feature names to values
        source: Which store served this request ("redis", "fallback", "cold_start")
        cache_hit: Whether the features came from cache (True) or were computed (False)
        latency_ms: Time taken to retrieve features in milliseconds
        entity_type: Type of entity ("user" or "movie")
        entity_id: The ID of the user or movie

    Example:
        >>> vec = FeatureVector(
        ...     features={"user_avg_rating": 3.5, "user_rating_count": 100},
        ...     source="redis",
        ...     cache_hit=True,
        ...     latency_ms=0.5
        ... )
        >>> print(f"Retrieved from {vec.source} in {vec.latency_ms}ms")
        Retrieved from redis in 0.5ms
    """

    features: Dict[str, float]
    source: str  # "redis", "fallback", "cold_start"
    cache_hit: bool
    latency_ms: float
    entity_type: str = ""  # "user" or "movie"
    entity_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "features": self.features,
            "source": self.source,
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
        }


@dataclass
class FeatureStoreStats:
    """
    Statistics about feature store operations.

    Used for monitoring and health checks.

    Attributes:
        total_requests: Total number of feature requests
        cache_hits: Number of requests served from cache
        cache_misses: Number of requests that missed cache
        errors: Number of errors encountered
        avg_latency_ms: Average latency in milliseconds
    """

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }


class FeatureStore(ABC):
    """
    Abstract interface for feature storage backends.

    All feature store implementations must inherit from this class and
    implement all abstract methods. This ensures consistent behavior
    whether using Redis, in-memory storage, or any other backend.

    Design Decisions:

    1. **Why return Optional[FeatureVector] instead of raising exceptions?**
       - Enables graceful fallback without try/except boilerplate
       - Caller can handle None cases with default values
       - More predictable behavior under partial failures

    2. **Why include batch methods?**
       - Network round-trips are expensive (especially for Redis)
       - Pipeline pattern enables significant performance gains
       - Ranking typically needs features for 100+ candidates

    3. **Why separate user/movie methods instead of generic get_features()?**
       - Different feature groups for users vs movies
       - Clear API contract
       - Type safety (user_id vs movie_id)

    Example Implementation:
        >>> class MyFeatureStore(FeatureStore):
        ...     def get_user_features(self, user_id: int) -> Optional[FeatureVector]:
        ...         # Fetch from your storage backend
        ...         return FeatureVector(features={...}, source="my_store", ...)
    """

    @abstractmethod
    def get_user_features(self, user_id: int) -> Optional[FeatureVector]:
        """
        Get all features for a single user.

        This retrieves both aggregation stats (rating count, avg rating, etc.)
        and demographic features (gender, age_group, occupation).

        Args:
            user_id: The user ID to fetch features for

        Returns:
            FeatureVector with user features, or None if:
            - User doesn't exist in the store
            - Store encountered an error (logged but not raised)

        Performance:
            Target: <1ms for Redis, <0.1ms for in-memory
        """
        pass

    @abstractmethod
    def get_movie_features(self, movie_id: int) -> Optional[FeatureVector]:
        """
        Get all features for a single movie.

        This retrieves both aggregation stats (rating count, avg rating, etc.)
        and genre features (18 binary genre flags).

        Args:
            movie_id: The movie ID to fetch features for

        Returns:
            FeatureVector with movie features, or None if:
            - Movie doesn't exist in the store
            - Store encountered an error (logged but not raised)

        Performance:
            Target: <1ms for Redis, <0.1ms for in-memory
        """
        pass

    @abstractmethod
    def get_batch_user_features(
        self, user_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get features for multiple users in one call.

        This is more efficient than calling get_user_features() in a loop
        because it can batch network requests (e.g., Redis pipeline).

        Args:
            user_ids: List of user IDs to fetch features for

        Returns:
            Dictionary mapping user_id -> FeatureVector (or None if missing)
            All requested user_ids will be keys in the result dict

        Performance:
            Target: <5ms for 100 users with Redis pipeline
        """
        pass

    @abstractmethod
    def get_batch_movie_features(
        self, movie_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get features for multiple movies in one call.

        This is the primary method used during ranking, as we typically
        need to score 100+ candidate movies for a single user.

        Args:
            movie_ids: List of movie IDs to fetch features for

        Returns:
            Dictionary mapping movie_id -> FeatureVector (or None if missing)
            All requested movie_ids will be keys in the result dict

        Performance:
            Target: <5ms for 100 movies with Redis pipeline
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the feature store is healthy and responsive.

        This method should:
        1. Verify connectivity to the underlying store
        2. Perform a simple read operation
        3. Return status and diagnostic information

        Returns:
            Dictionary with at least these keys:
            - "healthy": bool - Is the store operational?
            - "latency_ms": float - Time for health check operation
            - "message": str - Human-readable status message

        Used by:
            - /health endpoint to determine service health
            - Load balancers to route traffic
            - Monitoring systems for alerting
        """
        pass

    def get_stats(self) -> FeatureStoreStats:
        """
        Get operational statistics for this feature store.

        Default implementation returns empty stats. Subclasses should
        override to provide actual metrics.

        Returns:
            FeatureStoreStats with operational metrics
        """
        return FeatureStoreStats()

    def close(self) -> None:
        """
        Clean up resources (connections, pools, etc.).

        Default implementation does nothing. Subclasses should override
        if they hold resources that need cleanup.
        """
        pass
