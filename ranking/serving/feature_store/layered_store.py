"""
Layered feature store with circuit breaker pattern.

This module implements a layered feature store that combines a primary store
(Redis) with a fallback store (cold-start defaults). It includes a circuit
breaker to automatically switch to fallback when the primary store fails.

Circuit Breaker Pattern:
========================

The circuit breaker prevents cascading failures when Redis becomes unavailable.
Instead of repeatedly trying (and failing) to connect to Redis, the circuit
breaker opens and routes all traffic to the fallback store.

States:
1. CLOSED (normal): Requests go to primary store
   - If errors exceed threshold → transition to OPEN

2. OPEN (failing): Requests go directly to fallback store
   - After timeout → transition to HALF_OPEN

3. HALF_OPEN (recovery): Test requests go to primary
   - If success → transition to CLOSED
   - If failure → transition to OPEN

This pattern:
- Prevents wasting time on failed requests
- Allows the primary store time to recover
- Automatically recovers when primary becomes healthy
- Provides graceful degradation (service stays up)

Why This Matters in ML Systems:
===============================

In ML serving, latency budgets are tight (typically <50ms for ranking).
If Redis is slow or down:
- Without circuit breaker: Every request waits for timeout (e.g., 1s)
- With circuit breaker: Requests instantly use fallback (<1ms)

The fallback may give worse recommendations (no personalization), but
it's far better than timing out or returning errors.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .interface import FeatureStore, FeatureVector, FeatureStoreStats

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, primary store
    OPEN = "open"          # Failure state, fallback only
    HALF_OPEN = "half_open"  # Testing if primary recovered


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Successes needed in HALF_OPEN to close circuit
        timeout_seconds: How long to stay OPEN before trying HALF_OPEN
        failure_window_seconds: Time window for counting failures

    Tuning Guidelines:
        - failure_threshold: Lower = faster failover, higher = more resilient to blips
        - timeout_seconds: Lower = faster recovery attempts, higher = more time for fix
        - success_threshold: Higher = more confidence needed before recovery

    Production Defaults:
        The defaults below are conservative. In production you might tune:
        - failure_threshold=3 for faster failover
        - timeout_seconds=10 for faster recovery testing
    """
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    failure_window_seconds: float = 60.0


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation.

    This class tracks failures and manages state transitions. It's used
    by LayeredFeatureStore to decide whether to use primary or fallback.

    Thread Safety:
        All state modifications are protected by a lock to ensure correct
        behavior under concurrent requests.

    Example:
        >>> breaker = CircuitBreaker()
        >>> if breaker.allow_request():
        ...     try:
        ...         result = call_primary_store()
        ...         breaker.record_success()
        ...     except Exception:
        ...         breaker.record_failure()
        ...         result = call_fallback_store()
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker with given configuration."""
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"CircuitBreaker initialized: {self.config}")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state (thread-safe)."""
        with self._lock:
            return self._state

    def allow_request(self) -> bool:
        """
        Check if a request should go to the primary store.

        Returns:
            True if primary store should be tried
            False if fallback should be used directly

        Side Effects:
            May transition from OPEN to HALF_OPEN if timeout elapsed
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if time.time() - self._opened_at >= self.config.timeout_seconds:
                    logger.info("Circuit breaker: OPEN → HALF_OPEN (testing recovery)")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    return True
                return False

            # HALF_OPEN: Allow request to test recovery
            return True

    def record_success(self) -> None:
        """
        Record a successful request to primary store.

        May transition from HALF_OPEN to CLOSED if enough successes.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        f"Circuit breaker: HALF_OPEN → CLOSED "
                        f"(after {self._success_count} successes)"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed request to primary store.

        May transition to OPEN if failure threshold reached.
        """
        with self._lock:
            now = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN goes back to OPEN
                logger.warning("Circuit breaker: HALF_OPEN → OPEN (failure during recovery)")
                self._state = CircuitState.OPEN
                self._opened_at = now
                return

            # CLOSED state: count failures
            # Reset count if failure window expired
            if now - self._last_failure_time > self.config.failure_window_seconds:
                self._failure_count = 0

            self._failure_count += 1
            self._last_failure_time = now

            if self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit breaker: CLOSED → OPEN "
                    f"(after {self._failure_count} failures)"
                )
                self._state = CircuitState.OPEN
                self._opened_at = now

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status for monitoring."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            }


class LayeredFeatureStore(FeatureStore):
    """
    Feature store that combines primary and fallback stores with circuit breaker.

    This is the main feature store implementation for production use. It:
    1. Tries the primary store (Redis) first
    2. Falls back to secondary store on failure or missing data
    3. Uses circuit breaker to avoid hammering a failing primary

    Architecture:
        ┌─────────────────────────────────────────┐
        │         LayeredFeatureStore             │
        │  ┌─────────────┐   ┌─────────────────┐ │
        │  │ CircuitBreaker │   │               │ │
        │  └──────┬──────┘   │               │ │
        │         │          │               │ │
        │  ┌──────▼──────┐   │ Fallback      │ │
        │  │ Primary     │──▶│ Store         │ │
        │  │ (Redis)     │   │ (cold-start)  │ │
        │  └─────────────┘   └───────────────┘ │
        └─────────────────────────────────────────┘

    Behavior:
        - If circuit CLOSED: Try primary, fallback on error/missing
        - If circuit OPEN: Skip primary, use fallback directly
        - If circuit HALF_OPEN: Try primary to test recovery

    Example:
        >>> primary = RedisFeatureStore(host="localhost")
        >>> fallback = FallbackFeatureStore()
        >>> store = LayeredFeatureStore(primary, fallback)
        >>>
        >>> # Normal operation: tries Redis first
        >>> vec = store.get_user_features(123)
        >>> print(vec.source)  # "redis" or "fallback"
        >>>
        >>> # If Redis is down: automatically uses fallback
        >>> # If Redis recovers: automatically resumes using it
    """

    def __init__(
        self,
        primary: FeatureStore,
        fallback: FeatureStore,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize layered feature store.

        Args:
            primary: Primary store (typically Redis)
            fallback: Fallback store (typically cold-start defaults)
            circuit_config: Optional circuit breaker configuration
        """
        self.primary = primary
        self.fallback = fallback
        self.circuit_breaker = CircuitBreaker(circuit_config)

        # Track statistics
        self._primary_requests = 0
        self._fallback_requests = 0
        self._primary_successes = 0
        self._primary_failures = 0

        logger.info("LayeredFeatureStore initialized")

    def get_user_features(self, user_id: int) -> Optional[FeatureVector]:
        """
        Get user features, falling back if primary fails.

        Args:
            user_id: User ID to fetch features for

        Returns:
            FeatureVector with user features (never None for valid requests)
        """
        if self.circuit_breaker.allow_request():
            try:
                self._primary_requests += 1
                result = self.primary.get_user_features(user_id)

                if result is not None:
                    self._primary_successes += 1
                    self.circuit_breaker.record_success()
                    return result
                else:
                    # Primary returned None (entity not found)
                    # This is not a failure, but we need fallback
                    self._fallback_requests += 1
                    return self.fallback.get_user_features(user_id)

            except Exception as e:
                logger.warning(f"Primary store error for user {user_id}: {e}")
                self._primary_failures += 1
                self.circuit_breaker.record_failure()
                # Fall through to fallback

        # Circuit open or primary failed: use fallback
        self._fallback_requests += 1
        return self.fallback.get_user_features(user_id)

    def get_movie_features(self, movie_id: int) -> Optional[FeatureVector]:
        """
        Get movie features, falling back if primary fails.

        Args:
            movie_id: Movie ID to fetch features for

        Returns:
            FeatureVector with movie features
        """
        if self.circuit_breaker.allow_request():
            try:
                self._primary_requests += 1
                result = self.primary.get_movie_features(movie_id)

                if result is not None:
                    self._primary_successes += 1
                    self.circuit_breaker.record_success()
                    return result
                else:
                    self._fallback_requests += 1
                    return self.fallback.get_movie_features(movie_id)

            except Exception as e:
                logger.warning(f"Primary store error for movie {movie_id}: {e}")
                self._primary_failures += 1
                self.circuit_breaker.record_failure()

        self._fallback_requests += 1
        return self.fallback.get_movie_features(movie_id)

    def get_batch_user_features(
        self, user_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get batch user features with fallback for missing/failed lookups.

        Args:
            user_ids: List of user IDs

        Returns:
            Dictionary mapping user_id to FeatureVector
        """
        if self.circuit_breaker.allow_request():
            try:
                self._primary_requests += 1
                result = self.primary.get_batch_user_features(user_ids)

                # Check if any results are None (need fallback)
                missing_ids = [uid for uid, vec in result.items() if vec is None]

                if missing_ids:
                    self._fallback_requests += 1
                    fallback_results = self.fallback.get_batch_user_features(missing_ids)
                    for uid, vec in fallback_results.items():
                        result[uid] = vec

                # Record success if we got any results
                if any(vec is not None for vec in result.values()):
                    self._primary_successes += 1
                    self.circuit_breaker.record_success()

                return result

            except Exception as e:
                logger.warning(f"Primary store error for batch users: {e}")
                self._primary_failures += 1
                self.circuit_breaker.record_failure()

        # Circuit open or primary failed
        self._fallback_requests += 1
        return self.fallback.get_batch_user_features(user_ids)

    def get_batch_movie_features(
        self, movie_ids: List[int]
    ) -> Dict[int, Optional[FeatureVector]]:
        """
        Get batch movie features with fallback for missing/failed lookups.

        This is the critical path for ranking - called for 100+ movies per request.

        Args:
            movie_ids: List of movie IDs

        Returns:
            Dictionary mapping movie_id to FeatureVector
        """
        if self.circuit_breaker.allow_request():
            try:
                self._primary_requests += 1
                result = self.primary.get_batch_movie_features(movie_ids)

                # Fill in missing with fallback
                missing_ids = [mid for mid, vec in result.items() if vec is None]

                if missing_ids:
                    self._fallback_requests += 1
                    fallback_results = self.fallback.get_batch_movie_features(missing_ids)
                    for mid, vec in fallback_results.items():
                        result[mid] = vec

                if any(vec is not None for vec in result.values()):
                    self._primary_successes += 1
                    self.circuit_breaker.record_success()

                return result

            except Exception as e:
                logger.warning(f"Primary store error for batch movies: {e}")
                self._primary_failures += 1
                self.circuit_breaker.record_failure()

        self._fallback_requests += 1
        return self.fallback.get_batch_movie_features(movie_ids)

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of layered store.

        Returns healthy if either store is healthy (graceful degradation).

        Returns:
            Health status dictionary
        """
        primary_health = self.primary.health_check()
        fallback_health = self.fallback.health_check()
        circuit_status = self.circuit_breaker.get_status()

        # We're healthy if fallback is healthy (it always is)
        # Or if primary is healthy
        is_healthy = fallback_health.get("healthy", False) or primary_health.get("healthy", False)

        return {
            "healthy": is_healthy,
            "latency_ms": primary_health.get("latency_ms", 0.0),
            "message": self._build_health_message(primary_health, circuit_status),
            "type": "layered",
            "primary": primary_health,
            "fallback": fallback_health,
            "circuit_breaker": circuit_status,
        }

    def _build_health_message(
        self,
        primary_health: Dict[str, Any],
        circuit_status: Dict[str, Any],
    ) -> str:
        """Build human-readable health message."""
        if primary_health.get("healthy", False):
            return "Primary store healthy, operating normally"

        state = circuit_status.get("state", "unknown")
        if state == "open":
            return "Primary store unhealthy, circuit OPEN, using fallback"
        elif state == "half_open":
            return "Testing primary store recovery"
        else:
            return "Primary store unhealthy but circuit closed, will open soon"

    def get_stats(self) -> FeatureStoreStats:
        """Get combined statistics."""
        primary_stats = self.primary.get_stats()
        fallback_stats = self.fallback.get_stats()

        # Combine stats
        return FeatureStoreStats(
            total_requests=primary_stats.total_requests + fallback_stats.total_requests,
            cache_hits=primary_stats.cache_hits + fallback_stats.cache_hits,
            cache_misses=primary_stats.cache_misses + fallback_stats.cache_misses,
            errors=primary_stats.errors + fallback_stats.errors,
            avg_latency_ms=(primary_stats.avg_latency_ms + fallback_stats.avg_latency_ms) / 2,
        )

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for monitoring."""
        return {
            "primary_requests": self._primary_requests,
            "fallback_requests": self._fallback_requests,
            "primary_successes": self._primary_successes,
            "primary_failures": self._primary_failures,
            "primary_success_rate": (
                self._primary_successes / self._primary_requests
                if self._primary_requests > 0 else 0.0
            ),
            "fallback_rate": (
                self._fallback_requests / (self._primary_requests + self._fallback_requests)
                if (self._primary_requests + self._fallback_requests) > 0 else 0.0
            ),
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

    def close(self) -> None:
        """Close both stores."""
        self.primary.close()
        self.fallback.close()
