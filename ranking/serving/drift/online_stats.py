"""
Online statistics using Welford's algorithm.

This module provides streaming computation of statistics (mean, variance)
without storing all individual values. This is critical for production
drift detection where we may process millions of requests.

Welford's Algorithm:
===================

Traditional statistics require storing all values to compute mean and variance:
    mean = sum(x) / n
    var = sum((x - mean)^2) / n

Welford's algorithm computes these incrementally:
    1. For each new value x:
       - Update count: n += 1
       - Update mean: mean += (x - mean) / n
       - Update M2: M2 += (x - old_mean) * (x - new_mean)
    2. Variance = M2 / n

Benefits:
- O(1) space: Only stores count, mean, M2
- O(1) time per update: Constant time to add new value
- Numerically stable: Avoids catastrophic cancellation
- Incremental: Can compute at any time without reprocessing

This is essential for:
- Real-time feature monitoring
- Long-running services (no memory growth)
- Hourly/daily aggregations
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class WelfordAccumulator:
    """
    Single-feature accumulator using Welford's online algorithm.

    This accumulator maintains running statistics that can be updated
    incrementally without storing individual values.

    Attributes:
        count: Number of values seen
        mean: Running mean
        m2: Sum of squared deviations from mean (for variance)
        min_value: Minimum value seen
        max_value: Maximum value seen

    Example:
        >>> acc = WelfordAccumulator()
        >>> for value in [1, 2, 3, 4, 5]:
        ...     acc.update(value)
        >>> print(f"Mean: {acc.mean}, Std: {acc.std}")
        Mean: 3.0, Std: 1.414...
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared deviations
    min_value: float = float('inf')
    max_value: float = float('-inf')

    def update(self, value: float) -> None:
        """
        Update statistics with a new value.

        Uses Welford's online algorithm for numerical stability.

        Args:
            value: New value to incorporate
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def variance(self) -> float:
        """Compute variance from accumulated statistics."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Compute standard deviation."""
        return math.sqrt(self.variance)

    @property
    def sample_variance(self) -> float:
        """Compute sample variance (Bessel's correction)."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def sample_std(self) -> float:
        """Compute sample standard deviation."""
        return math.sqrt(self.sample_variance)

    def merge(self, other: "WelfordAccumulator") -> "WelfordAccumulator":
        """
        Merge two accumulators (parallel/batch processing).

        This allows combining statistics from different time periods
        or different servers without losing accuracy.

        Args:
            other: Another accumulator to merge

        Returns:
            New merged accumulator
        """
        if self.count == 0:
            return WelfordAccumulator(
                count=other.count,
                mean=other.mean,
                m2=other.m2,
                min_value=other.min_value,
                max_value=other.max_value,
            )
        if other.count == 0:
            return WelfordAccumulator(
                count=self.count,
                mean=self.mean,
                m2=self.m2,
                min_value=self.min_value,
                max_value=self.max_value,
            )

        combined_count = self.count + other.count
        delta = other.mean - self.mean
        combined_mean = (
            self.mean * self.count + other.mean * other.count
        ) / combined_count
        combined_m2 = (
            self.m2 + other.m2 +
            delta * delta * self.count * other.count / combined_count
        )

        return WelfordAccumulator(
            count=combined_count,
            mean=combined_mean,
            m2=combined_m2,
            min_value=min(self.min_value, other.min_value),
            max_value=max(self.max_value, other.max_value),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "min_value": self.min_value if self.min_value != float('inf') else None,
            "max_value": self.max_value if self.max_value != float('-inf') else None,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WelfordAccumulator":
        """Create from dictionary."""
        return cls(
            count=data.get("count", 0),
            mean=data.get("mean", 0.0),
            m2=data.get("m2", 0.0),
            min_value=data.get("min_value") or float('inf'),
            max_value=data.get("max_value") or float('-inf'),
        )


class OnlineStatistics:
    """
    Thread-safe online statistics tracker for multiple features.

    This class maintains Welford accumulators for each feature and provides
    thread-safe updates for concurrent request processing.

    Features:
    - Thread-safe updates via RLock
    - Per-feature accumulator management
    - Snapshot export for drift computation
    - Reset capability for time windowing

    Example:
        >>> stats = OnlineStatistics()
        >>> stats.update("user_avg_rating", 3.5)
        >>> stats.update("user_avg_rating", 4.0)
        >>> snapshot = stats.get_snapshot()
        >>> print(snapshot["user_avg_rating"]["mean"])
        3.75
    """

    def __init__(self):
        """Initialize empty online statistics tracker."""
        self._accumulators: Dict[str, WelfordAccumulator] = {}
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._last_reset = time.time()

    def update(self, feature_name: str, value: float) -> None:
        """
        Update statistics for a feature with a new value.

        Thread-safe operation.

        Args:
            feature_name: Name of the feature
            value: New value to incorporate
        """
        with self._lock:
            if feature_name not in self._accumulators:
                self._accumulators[feature_name] = WelfordAccumulator()
            self._accumulators[feature_name].update(value)

    def update_batch(self, features: Dict[str, float]) -> None:
        """
        Update multiple features at once.

        More efficient than multiple single updates due to single lock acquisition.

        Args:
            features: Dictionary mapping feature names to values
        """
        with self._lock:
            for name, value in features.items():
                if name not in self._accumulators:
                    self._accumulators[name] = WelfordAccumulator()
                self._accumulators[name].update(value)

    def get_accumulator(self, feature_name: str) -> Optional[WelfordAccumulator]:
        """Get the accumulator for a specific feature."""
        with self._lock:
            return self._accumulators.get(feature_name)

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a snapshot of all current statistics.

        Returns a copy that can be safely used outside the lock.

        Returns:
            Dictionary mapping feature names to their statistics
        """
        with self._lock:
            return {
                name: acc.to_dict()
                for name, acc in self._accumulators.items()
            }

    def get_feature_names(self) -> List[str]:
        """Get list of tracked feature names."""
        with self._lock:
            return list(self._accumulators.keys())

    def get_total_samples(self) -> int:
        """Get total number of samples across all features."""
        with self._lock:
            if not self._accumulators:
                return 0
            # All features should have same count if updated together
            return max(acc.count for acc in self._accumulators.values())

    def reset(self) -> Dict[str, Dict[str, Any]]:
        """
        Reset all accumulators and return final snapshot.

        Used for time-windowed statistics (e.g., hourly aggregations).

        Returns:
            Final snapshot before reset
        """
        with self._lock:
            snapshot = self.get_snapshot()
            self._accumulators.clear()
            self._last_reset = time.time()
            return snapshot

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the statistics tracker."""
        with self._lock:
            return {
                "started_at": self._started_at,
                "last_reset": self._last_reset,
                "elapsed_seconds": time.time() - self._last_reset,
                "num_features": len(self._accumulators),
                "total_samples": self.get_total_samples(),
            }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        with self._lock:
            data = {
                "metadata": self.get_metadata(),
                "features": self.get_snapshot(),
            }
            return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "OnlineStatistics":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        instance = cls()

        for name, acc_data in data.get("features", {}).items():
            instance._accumulators[name] = WelfordAccumulator.from_dict(acc_data)

        metadata = data.get("metadata", {})
        instance._started_at = metadata.get("started_at", time.time())
        instance._last_reset = metadata.get("last_reset", time.time())

        return instance
