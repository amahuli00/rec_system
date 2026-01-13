"""
Drift detector that combines baseline and online statistics.

This module provides the DriftDetector class that orchestrates drift detection
by comparing baseline statistics against online statistics using various metrics.

Architecture:
=============

              Request Features
                    │
                    ▼
            ┌───────────────┐
            │ OnlineStats   │  ← Update with each request
            │ (Welford)     │
            └───────┬───────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    ┌─────────────┐    ┌─────────────┐
    │ Baseline    │    │ Scheduler   │  ← Hourly comparison
    │ Statistics  │    │ (periodic)  │
    └──────┬──────┘    └──────┬──────┘
           │                  │
           └────────┬─────────┘
                    ▼
            ┌───────────────┐
            │ DriftMetrics  │  ← Compute Z-score, KL, PSI
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ DriftReport   │  ← Summary with alerts
            └───────────────┘

Usage:
======

1. Initialize with baseline statistics
2. Log features with each request
3. Periodically check for drift (or use scheduler)
4. Export reports for monitoring

Example:
    >>> detector = DriftDetector(baseline)
    >>> detector.log_features({"user_avg_rating": 3.5, "movie_avg_rating": 4.0})
    >>> report = detector.check_drift()
    >>> if report.has_alerts:
    ...     print(f"Drift detected: {report.alerting_features}")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .baseline import BaselineStatistics
from .online_stats import OnlineStatistics
from .metrics import DriftMetrics, DriftResult, DriftSeverity

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """
    Summary report of drift detection results.

    Attributes:
        timestamp: When the report was generated
        num_features: Number of features analyzed
        num_ok: Features with no drift
        num_warning: Features with warning-level drift
        num_alert: Features with alert-level drift
        results: Per-feature drift results
        online_samples: Number of online samples used
        metadata: Additional diagnostic information
    """
    timestamp: str
    num_features: int
    num_ok: int
    num_warning: int
    num_alert: int
    results: List[DriftResult]
    online_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_alerts(self) -> bool:
        """Check if any features have alert-level drift."""
        return self.num_alert > 0

    @property
    def has_warnings(self) -> bool:
        """Check if any features have warning-level drift."""
        return self.num_warning > 0

    @property
    def alerting_features(self) -> List[str]:
        """Get list of features with alert-level drift."""
        return [r.feature_name for r in self.results if r.severity == DriftSeverity.ALERT]

    @property
    def warning_features(self) -> List[str]:
        """Get list of features with warning-level drift."""
        return [r.feature_name for r in self.results if r.severity == DriftSeverity.WARNING]

    @property
    def overall_status(self) -> str:
        """Get overall status string."""
        if self.num_alert > 0:
            return "alert"
        if self.num_warning > 0:
            return "warning"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "status": self.overall_status,
            "num_features": self.num_features,
            "num_ok": self.num_ok,
            "num_warning": self.num_warning,
            "num_alert": self.num_alert,
            "online_samples": self.online_samples,
            "alerting_features": self.alerting_features,
            "warning_features": self.warning_features,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DriftDetector:
    """
    Main drift detection orchestrator.

    This class coordinates between baseline statistics, online statistics,
    and drift metrics to provide comprehensive drift detection.

    Features:
    - Log feature values from production requests
    - Compute drift on demand or on schedule
    - Generate summary reports with alerts
    - Support for feature subsetting (check specific features only)

    Example:
        >>> # Initialize with baseline
        >>> baseline = BaselineStatistics.load("baseline.json")
        >>> detector = DriftDetector(baseline)
        >>>
        >>> # Log features during request processing
        >>> features = {"user_avg_rating": 3.5, "movie_avg_rating": 4.0}
        >>> detector.log_features(features)
        >>>
        >>> # Check drift periodically
        >>> report = detector.check_drift()
        >>> print(f"Status: {report.overall_status}")
        >>> if report.has_alerts:
        ...     print(f"Alerting features: {report.alerting_features}")
    """

    # Minimum samples before computing drift
    MIN_SAMPLES_FOR_DRIFT = 100

    def __init__(
        self,
        baseline: BaselineStatistics,
        feature_subset: Optional[List[str]] = None,
    ):
        """
        Initialize drift detector.

        Args:
            baseline: BaselineStatistics from training data
            feature_subset: Optional list of features to monitor (default: all)
        """
        self.baseline = baseline
        self.online_stats = OnlineStatistics()

        # Determine which features to monitor
        if feature_subset:
            self.monitored_features = [
                f for f in feature_subset
                if f in baseline.features
            ]
        else:
            self.monitored_features = list(baseline.features.keys())

        self._last_report: Optional[DriftReport] = None
        self._report_history: List[DriftReport] = []

        logger.info(
            f"DriftDetector initialized with {len(self.monitored_features)} features"
        )

    def log_features(self, features: Dict[str, float]) -> None:
        """
        Log feature values from a production request.

        This should be called for each recommendation request to accumulate
        online statistics for drift detection.

        Args:
            features: Dictionary mapping feature names to values
        """
        # Only log monitored features
        filtered = {
            k: v for k, v in features.items()
            if k in self.monitored_features
        }
        self.online_stats.update_batch(filtered)

    def check_drift(
        self,
        reset_after: bool = False,
        features: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Compute drift metrics and generate a report.

        Args:
            reset_after: Reset online statistics after checking (for time windows)
            features: Optional subset of features to check (default: all monitored)

        Returns:
            DriftReport with per-feature results and summary
        """
        start_time = time.time()

        # Determine features to check
        features_to_check = features or self.monitored_features

        # Get online statistics snapshot
        online_snapshot = self.online_stats.get_snapshot()
        online_samples = self.online_stats.get_total_samples()

        # Check if we have enough samples
        if online_samples < self.MIN_SAMPLES_FOR_DRIFT:
            logger.warning(
                f"Insufficient samples for drift detection: "
                f"{online_samples} < {self.MIN_SAMPLES_FOR_DRIFT}"
            )

        # Compute drift for each feature
        results: List[DriftResult] = []

        for feature_name in features_to_check:
            baseline_feat = self.baseline.get_feature(feature_name)
            online_feat = online_snapshot.get(feature_name)

            if baseline_feat is None:
                logger.warning(f"No baseline for feature {feature_name}")
                continue

            if online_feat is None or online_feat.get("count", 0) == 0:
                logger.debug(f"No online data for feature {feature_name}")
                continue

            # Compute drift metrics
            result = DriftMetrics.compute_all(
                feature_name=feature_name,
                baseline_mean=baseline_feat.mean,
                baseline_std=baseline_feat.std,
                online_mean=online_feat["mean"],
                online_std=online_feat["std"],
                baseline_proportions=baseline_feat.histogram_counts,
                online_proportions=None,  # Would need histogram from online
            )
            results.append(result)

        # Count severities
        num_ok = sum(1 for r in results if r.severity == DriftSeverity.OK)
        num_warning = sum(1 for r in results if r.severity == DriftSeverity.WARNING)
        num_alert = sum(1 for r in results if r.severity == DriftSeverity.ALERT)

        # Build report
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            num_features=len(results),
            num_ok=num_ok,
            num_warning=num_warning,
            num_alert=num_alert,
            results=results,
            online_samples=online_samples,
            metadata={
                "computation_time_ms": (time.time() - start_time) * 1000,
                "min_samples_threshold": self.MIN_SAMPLES_FOR_DRIFT,
            },
        )

        # Store report
        self._last_report = report
        self._report_history.append(report)

        # Log summary
        if report.has_alerts:
            logger.warning(
                f"Drift ALERT: {report.num_alert} features "
                f"({', '.join(report.alerting_features)})"
            )
        elif report.has_warnings:
            logger.info(
                f"Drift WARNING: {report.num_warning} features "
                f"({', '.join(report.warning_features)})"
            )
        else:
            logger.debug(f"Drift check OK: {report.num_features} features stable")

        # Reset if requested (for time-windowed checks)
        if reset_after:
            self.online_stats.reset()
            logger.debug("Reset online statistics after drift check")

        return report

    def get_last_report(self) -> Optional[DriftReport]:
        """Get the most recent drift report."""
        return self._last_report

    def get_report_history(self, limit: int = 10) -> List[DriftReport]:
        """Get recent drift reports."""
        return self._report_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status for API endpoints."""
        online_meta = self.online_stats.get_metadata()

        status = {
            "monitored_features": len(self.monitored_features),
            "online_samples": online_meta["total_samples"],
            "elapsed_seconds": online_meta["elapsed_seconds"],
            "last_report": (
                self._last_report.to_dict() if self._last_report else None
            ),
        }

        if self._last_report:
            status["current_status"] = self._last_report.overall_status
        else:
            status["current_status"] = "not_checked"

        return status

    def reset(self) -> None:
        """Reset online statistics and report history."""
        self.online_stats.reset()
        self._last_report = None
        self._report_history.clear()
        logger.info("DriftDetector reset")
