"""
Feature Store Package

Provides abstraction for feature storage and retrieval with pluggable backends.

Components:
- FeatureStore: Abstract interface for feature stores
- FeatureVector: Data class for feature retrieval results
- RedisFeatureStore: Redis-backed implementation with connection pooling
- FallbackFeatureStore: In-memory fallback using cold-start defaults
- LayeredFeatureStore: Primary + fallback with circuit breaker pattern
"""

from .interface import FeatureStore, FeatureVector, FeatureStoreStats
from .redis_store import RedisFeatureStore
from .fallback_store import FallbackFeatureStore
from .layered_store import LayeredFeatureStore, CircuitBreakerConfig, CircuitBreaker

__all__ = [
    # Interface
    "FeatureStore",
    "FeatureVector",
    "FeatureStoreStats",
    # Implementations
    "RedisFeatureStore",
    "FallbackFeatureStore",
    "LayeredFeatureStore",
    # Circuit Breaker
    "CircuitBreakerConfig",
    "CircuitBreaker",
]
