"""
Redis-based feature store implementation.

This module provides a Redis implementation of the FeatureStore interface.
It's designed for production use with features like:
- Connection pooling for concurrent access
- Pipeline pattern for batch operations
- Graceful error handling with logging
- Observability metrics

Redis Data Model:
    User features are stored in two hashes per user:
    - features:user:{user_id}:stats       → {user_rating_count, user_avg_rating, ...}
    - features:user:{user_id}:demographics → {gender, age_group, occupation}

    Movie features are stored in two hashes per movie:
    - features:movie:{movie_id}:stats     → {movie_rating_count, movie_avg_rating, ...}
    - features:movie:{movie_id}:genres    → {genre_action, genre_comedy, ...}

Why Redis HASH?
    - Field-level access (no need to deserialize entire object)
    - Memory efficient for small feature sets
    - HGETALL retrieves all fields in one operation
    - HMGET for selective field retrieval

Performance Considerations:
    - Connection pooling: Reuse connections across requests
    - Pipeline: Batch multiple commands into single round-trip
    - Decode responses: Handle byte/string conversion automatically
    - Timeouts: Fail fast on slow responses
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .interface import FeatureStore, FeatureStoreStats, FeatureVector

logger = logging.getLogger(__name__)

# Try to import redis, but handle gracefully if not installed
try:
    from redis import ConnectionPool, Redis, RedisError
    from redis.retry import Retry
    from redis.backoff import ExponentialBackoff

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed. RedisFeatureStore will not be available.")


class RedisFeatureStore(FeatureStore):
    """
    Redis implementation of the feature store.

    This class provides high-performance feature retrieval using Redis.
    It uses connection pooling and the pipeline pattern for efficiency.

    Architecture:
        ┌─────────────────────┐
        │  RedisFeatureStore  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   ConnectionPool    │
        │  (max_connections)  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │    Redis Server     │
        └─────────────────────┘

    Usage:
        >>> store = RedisFeatureStore(host="localhost", port=6379)
        >>> features = store.get_user_features(user_id=123)
        >>> if features:
        ...     print(f"User avg rating: {features.features['user_avg_rating']}")

    Configuration:
        - host: Redis server hostname
        - port: Redis server port (default: 6379)
        - db: Redis database number (default: 0)
        - password: Optional authentication password
        - max_connections: Maximum pool connections (default: 50)
        - socket_timeout: Timeout for socket operations (default: 0.1s)
        - socket_connect_timeout: Timeout for connection (default: 0.5s)
    """

    # Redis key prefixes
    USER_STATS_KEY = "features:user:{user_id}:stats"
    USER_DEMOGRAPHICS_KEY = "features:user:{user_id}:demographics"
    MOVIE_STATS_KEY = "features:movie:{movie_id}:stats"
    MOVIE_GENRES_KEY = "features:movie:{movie_id}:genres"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: float = 0.1,  # 100ms timeout for operations
        socket_connect_timeout: float = 0.5,  # 500ms timeout for connection
        retry_on_error: bool = True,
    ):
        """
        Initialize the Redis feature store.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number (0-15)
            password: Optional authentication password
            max_connections: Maximum connections in pool
            socket_timeout: Timeout for read/write operations (seconds)
            socket_connect_timeout: Timeout for initial connection (seconds)
            retry_on_error: Whether to retry on transient errors

        Raises:
            ImportError: If redis package is not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for RedisFeatureStore. "
                "Install it with: pip install redis"
            )

        self.host = host
        self.port = port
        self.db = db

        # Create connection pool
        # Why connection pool?
        # - Avoids creating new TCP connections per request (expensive)
        # - Limits concurrent connections to prevent overwhelming Redis
        # - Thread-safe for concurrent access
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True,  # Auto-decode bytes to strings
        )

        # Create retry strategy for transient failures
        # Why retry?
        # - Network blips shouldn't cause request failures
        # - Exponential backoff prevents thundering herd
        if retry_on_error:
            retry = Retry(ExponentialBackoff(), retries=2)
            self.client = Redis(connection_pool=self.pool, retry=retry)
        else:
            self.client = Redis(connection_pool=self.pool)

        # Stats tracking
        self._stats = FeatureStoreStats()
        self._total_latency_ms = 0.0

        logger.info(f"RedisFeatureStore initialized: {host}:{port}/{db}")

    def get_user_features(self, user_id: int) -> Optional[FeatureVector]:
        """
        Get all features for a single user.

        Uses Redis pipeline to fetch stats and demographics in one round-trip.

        Args:
            user_id: The user ID to fetch features for

        Returns:
            FeatureVector with user features, or None if not found/error
        """
        start_time = time.time()
        self._stats.total_requests += 1

        try:
            # Use pipeline to fetch both hashes in one round-trip
            # Why pipeline?
            # - Without: 2 round-trips (stats, demographics)
            # - With: 1 round-trip for both
            # - 50% latency reduction
            pipe = self.client.pipeline(transaction=False)

            stats_key = self.USER_STATS_KEY.format(user_id=user_id)
            demo_key = self.USER_DEMOGRAPHICS_KEY.format(user_id=user_id)

            pipe.hgetall(stats_key)
            pipe.hgetall(demo_key)

            results = pipe.execute()

            stats_data = results[0]
            demo_data = results[1]

            # Check if user exists (at least stats should be present)
            if not stats_data:
                self._stats.cache_misses += 1
                latency_ms = (time.time() - start_time) * 1000
                self._update_latency(latency_ms)
                logger.debug(f"User {user_id} not found in Redis")
                return None

            # Merge and convert to floats
            features = {}
            for key, value in stats_data.items():
                features[key] = float(value)
            for key, value in demo_data.items():
                features[key] = float(value)

            self._stats.cache_hits += 1
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)

            return FeatureVector(
                features=features,
                source="redis",
                cache_hit=True,
                latency_ms=latency_ms,
                entity_type="user",
                entity_id=user_id,
            )

        except RedisError as e:
            self._stats.errors += 1
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)
            logger.warning(f"Redis error fetching user {user_id}: {e}")
            return None

    def get_movie_features(self, movie_id: int) -> Optional[FeatureVector]:
        """
        Get all features for a single movie.

        Uses Redis pipeline to fetch stats and genres in one round-trip.

        Args:
            movie_id: The movie ID to fetch features for

        Returns:
            FeatureVector with movie features, or None if not found/error
        """
        start_time = time.time()
        self._stats.total_requests += 1

        try:
            pipe = self.client.pipeline(transaction=False)

            stats_key = self.MOVIE_STATS_KEY.format(movie_id=movie_id)
            genres_key = self.MOVIE_GENRES_KEY.format(movie_id=movie_id)

            pipe.hgetall(stats_key)
            pipe.hgetall(genres_key)

            results = pipe.execute()

            stats_data = results[0]
            genres_data = results[1]

            # Check if movie exists
            if not stats_data:
                self._stats.cache_misses += 1
                latency_ms = (time.time() - start_time) * 1000
                self._update_latency(latency_ms)
                logger.debug(f"Movie {movie_id} not found in Redis")
                return None

            # Merge and convert to floats
            features = {}
            for key, value in stats_data.items():
                features[key] = float(value)
            for key, value in genres_data.items():
                features[key] = float(value)

            self._stats.cache_hits += 1
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)

            return FeatureVector(
                features=features,
                source="redis",
                cache_hit=True,
                latency_ms=latency_ms,
                entity_type="movie",
                entity_id=movie_id,
            )

        except RedisError as e:
            self._stats.errors += 1
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)
            logger.warning(f"Redis error fetching movie {movie_id}: {e}")
            return None

    def get_batch_user_features(
        self, user_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get features for multiple users in one call.

        Uses Redis pipeline to batch all requests into a single round-trip.
        This is significantly faster than calling get_user_features() in a loop.

        Performance Example:
            10 users without pipeline: ~10ms (10 round-trips * 1ms)
            10 users with pipeline: ~1ms (1 round-trip)

        Args:
            user_ids: List of user IDs to fetch

        Returns:
            Dictionary mapping user_id -> FeatureVector (or None)
        """
        if not user_ids:
            return {}

        start_time = time.time()
        self._stats.total_requests += len(user_ids)
        results: Dict[int, Optional[FeatureVector]] = {}

        try:
            pipe = self.client.pipeline(transaction=False)

            # Queue all requests
            # Order: [user1_stats, user1_demo, user2_stats, user2_demo, ...]
            for user_id in user_ids:
                stats_key = self.USER_STATS_KEY.format(user_id=user_id)
                demo_key = self.USER_DEMOGRAPHICS_KEY.format(user_id=user_id)
                pipe.hgetall(stats_key)
                pipe.hgetall(demo_key)

            # Execute all at once
            raw_results = pipe.execute()

            # Parse results (every 2 items = stats + demo for one user)
            for i, user_id in enumerate(user_ids):
                stats_data = raw_results[i * 2]
                demo_data = raw_results[i * 2 + 1]

                if stats_data:
                    features = {}
                    for key, value in stats_data.items():
                        features[key] = float(value)
                    for key, value in demo_data.items():
                        features[key] = float(value)

                    self._stats.cache_hits += 1
                    results[user_id] = FeatureVector(
                        features=features,
                        source="redis",
                        cache_hit=True,
                        latency_ms=0,  # Will set total latency below
                        entity_type="user",
                        entity_id=user_id,
                    )
                else:
                    self._stats.cache_misses += 1
                    results[user_id] = None

            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)

            # Update latency for all successful results
            for fv in results.values():
                if fv is not None:
                    fv.latency_ms = latency_ms / len(user_ids)

            return results

        except RedisError as e:
            self._stats.errors += len(user_ids)
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)
            logger.warning(f"Redis error in batch user fetch: {e}")
            return {user_id: None for user_id in user_ids}

    def get_batch_movie_features(
        self, movie_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get features for multiple movies in one call.

        This is the primary method used during ranking. For 100 candidate
        movies, this reduces Redis latency from ~100ms to ~5ms.

        Args:
            movie_ids: List of movie IDs to fetch

        Returns:
            Dictionary mapping movie_id -> FeatureVector (or None)
        """
        if not movie_ids:
            return {}

        start_time = time.time()
        self._stats.total_requests += len(movie_ids)
        results: Dict[int, Optional[FeatureVector]] = {}

        try:
            pipe = self.client.pipeline(transaction=False)

            # Queue all requests
            for movie_id in movie_ids:
                stats_key = self.MOVIE_STATS_KEY.format(movie_id=movie_id)
                genres_key = self.MOVIE_GENRES_KEY.format(movie_id=movie_id)
                pipe.hgetall(stats_key)
                pipe.hgetall(genres_key)

            # Execute all at once
            raw_results = pipe.execute()

            # Parse results
            for i, movie_id in enumerate(movie_ids):
                stats_data = raw_results[i * 2]
                genres_data = raw_results[i * 2 + 1]

                if stats_data:
                    features = {}
                    for key, value in stats_data.items():
                        features[key] = float(value)
                    for key, value in genres_data.items():
                        features[key] = float(value)

                    self._stats.cache_hits += 1
                    results[movie_id] = FeatureVector(
                        features=features,
                        source="redis",
                        cache_hit=True,
                        latency_ms=0,
                        entity_type="movie",
                        entity_id=movie_id,
                    )
                else:
                    self._stats.cache_misses += 1
                    results[movie_id] = None

            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)

            # Update latency for all successful results
            for fv in results.values():
                if fv is not None:
                    fv.latency_ms = latency_ms / len(movie_ids)

            return results

        except RedisError as e:
            self._stats.errors += len(movie_ids)
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency(latency_ms)
            logger.warning(f"Redis error in batch movie fetch: {e}")
            return {movie_id: None for movie_id in movie_ids}

    def health_check(self) -> Dict[str, Any]:
        """
        Check if Redis is healthy and responsive.

        Performs a PING command and measures latency.

        Returns:
            Dictionary with health status and diagnostics
        """
        start_time = time.time()

        try:
            # PING is a simple command that verifies connectivity
            response = self.client.ping()
            latency_ms = (time.time() - start_time) * 1000

            if response:
                return {
                    "healthy": True,
                    "latency_ms": latency_ms,
                    "message": "Redis is responsive",
                    "host": self.host,
                    "port": self.port,
                    "db": self.db,
                }
            else:
                return {
                    "healthy": False,
                    "latency_ms": latency_ms,
                    "message": "Redis PING returned unexpected response",
                    "host": self.host,
                    "port": self.port,
                    "db": self.db,
                }

        except RedisError as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "healthy": False,
                "latency_ms": latency_ms,
                "message": f"Redis error: {e}",
                "host": self.host,
                "port": self.port,
                "db": self.db,
            }

    def get_stats(self) -> FeatureStoreStats:
        """Get operational statistics."""
        return self._stats

    def close(self) -> None:
        """Close the connection pool."""
        try:
            self.pool.disconnect()
            logger.info("RedisFeatureStore connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing connection pool: {e}")

    def _update_latency(self, latency_ms: float) -> None:
        """Update average latency tracking."""
        self._total_latency_ms += latency_ms
        if self._stats.total_requests > 0:
            self._stats.avg_latency_ms = (
                self._total_latency_ms / self._stats.total_requests
            )

    # =========================================================================
    # Write Methods (for migration and testing)
    # =========================================================================

    def set_user_features(
        self,
        user_id: int,
        stats: Dict[str, float],
        demographics: Dict[str, float],
    ) -> bool:
        """
        Store user features in Redis.

        Used during migration and testing. In production, features are
        typically loaded from batch pipelines.

        Args:
            user_id: The user ID
            stats: User aggregation stats (rating_count, avg_rating, etc.)
            demographics: User demographics (gender, age_group, occupation)

        Returns:
            True if successful, False otherwise
        """
        try:
            pipe = self.client.pipeline(transaction=True)

            stats_key = self.USER_STATS_KEY.format(user_id=user_id)
            demo_key = self.USER_DEMOGRAPHICS_KEY.format(user_id=user_id)

            if stats:
                pipe.hset(stats_key, mapping=stats)
            if demographics:
                pipe.hset(demo_key, mapping=demographics)

            pipe.execute()
            return True

        except RedisError as e:
            logger.error(f"Error setting user features for {user_id}: {e}")
            return False

    def set_movie_features(
        self,
        movie_id: int,
        stats: Dict[str, float],
        genres: Dict[str, float],
    ) -> bool:
        """
        Store movie features in Redis.

        Args:
            movie_id: The movie ID
            stats: Movie aggregation stats
            genres: Movie genre flags

        Returns:
            True if successful, False otherwise
        """
        try:
            pipe = self.client.pipeline(transaction=True)

            stats_key = self.MOVIE_STATS_KEY.format(movie_id=movie_id)
            genres_key = self.MOVIE_GENRES_KEY.format(movie_id=movie_id)

            if stats:
                pipe.hset(stats_key, mapping=stats)
            if genres:
                pipe.hset(genres_key, mapping=genres)

            pipe.execute()
            return True

        except RedisError as e:
            logger.error(f"Error setting movie features for {movie_id}: {e}")
            return False

    def get_key_count(self) -> int:
        """Get total number of keys in the database."""
        try:
            return self.client.dbsize()
        except RedisError as e:
            logger.error(f"Error getting key count: {e}")
            return -1

    def flush_db(self) -> bool:
        """
        Delete all keys in the current database.

        WARNING: This is destructive! Only use in development/testing.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.flushdb()
            logger.warning(f"Flushed Redis database {self.db}")
            return True
        except RedisError as e:
            logger.error(f"Error flushing database: {e}")
            return False
