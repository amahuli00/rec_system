"""
Drift detection metrics.

This module provides mathematical functions for computing drift between
baseline (training) and online (production) distributions.

Metrics Implemented:
====================

1. **Mean Shift (Z-score)**
   - Simple and interpretable
   - Measures how many standard deviations the mean has shifted
   - Formula: z = |online_mean - baseline_mean| / baseline_std
   - Threshold: z > 3 indicates significant drift

2. **KL Divergence (Gaussian approximation)**
   - Information-theoretic measure
   - Quantifies how much information is lost using baseline instead of online
   - Formula: KL(P||Q) = log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²)/(2σ_Q²) - 0.5
   - Note: Asymmetric - KL(P||Q) ≠ KL(Q||P)
   - Threshold: KL > 0.5 indicates significant drift

3. **PSI (Population Stability Index)**
   - Industry standard in finance/credit scoring
   - Compares distributions using histogram bins
   - Formula: PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
   - Thresholds:
     - PSI < 0.1: No significant change
     - 0.1 ≤ PSI < 0.2: Moderate change
     - PSI ≥ 0.2: Significant change

Why Multiple Metrics?
====================

Each metric captures different aspects of drift:
- Mean shift: Detects location changes (simple but misses variance changes)
- KL divergence: Detects both location and scale changes (assumes Gaussian)
- PSI: Distribution-free, captures any shape change (needs histogram data)

Using multiple metrics provides more robust drift detection and helps
diagnose the type of drift occurring.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity levels for drift alerts."""
    OK = "ok"
    WARNING = "warning"
    ALERT = "alert"


@dataclass
class DriftResult:
    """
    Result of drift detection for a single feature.

    Attributes:
        feature_name: Name of the feature
        mean_shift_z: Z-score of mean shift
        kl_divergence: KL divergence (Gaussian approximation)
        psi: Population Stability Index
        severity: Overall drift severity
        details: Additional diagnostic information
    """
    feature_name: str
    mean_shift_z: float
    kl_divergence: Optional[float]
    psi: Optional[float]
    severity: DriftSeverity
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_name": self.feature_name,
            "mean_shift_z": self.mean_shift_z,
            "kl_divergence": self.kl_divergence,
            "psi": self.psi,
            "severity": self.severity.value,
            "details": self.details,
        }


class DriftMetrics:
    """
    Computes drift metrics between baseline and online statistics.

    This class provides static methods for computing various drift metrics.
    It can be used to compare training data statistics against production
    data statistics to detect feature drift.

    Example:
        >>> baseline_mean, baseline_std = 3.5, 1.0
        >>> online_mean, online_std = 4.2, 1.1
        >>>
        >>> z = DriftMetrics.mean_shift_zscore(
        ...     baseline_mean, baseline_std, online_mean
        ... )
        >>> print(f"Z-score: {z:.2f}")  # 0.70 - within normal range
        >>>
        >>> kl = DriftMetrics.kl_divergence_gaussian(
        ...     baseline_mean, baseline_std, online_mean, online_std
        ... )
        >>> print(f"KL divergence: {kl:.3f}")
    """

    # Thresholds for severity classification
    MEAN_SHIFT_WARNING_THRESHOLD = 2.0  # 2 std deviations
    MEAN_SHIFT_ALERT_THRESHOLD = 3.0    # 3 std deviations
    KL_WARNING_THRESHOLD = 0.3
    KL_ALERT_THRESHOLD = 0.5
    PSI_WARNING_THRESHOLD = 0.1
    PSI_ALERT_THRESHOLD = 0.2

    # Small epsilon to avoid division by zero and log(0)
    EPSILON = 1e-10

    @staticmethod
    def mean_shift_zscore(
        baseline_mean: float,
        baseline_std: float,
        online_mean: float,
    ) -> float:
        """
        Compute Z-score of mean shift.

        Measures how many baseline standard deviations the online mean
        has shifted from the baseline mean.

        Args:
            baseline_mean: Mean from training data
            baseline_std: Standard deviation from training data
            online_mean: Mean from production data

        Returns:
            Absolute Z-score of the mean shift

        Example:
            >>> z = DriftMetrics.mean_shift_zscore(3.5, 1.0, 4.2)
            >>> print(f"Mean shifted by {z:.1f} standard deviations")
            Mean shifted by 0.7 standard deviations
        """
        if baseline_std < DriftMetrics.EPSILON:
            # Constant feature - any change is significant
            if abs(online_mean - baseline_mean) > DriftMetrics.EPSILON:
                return float('inf')
            return 0.0

        return abs(online_mean - baseline_mean) / baseline_std

    @staticmethod
    def kl_divergence_gaussian(
        p_mean: float,
        p_std: float,
        q_mean: float,
        q_std: float,
    ) -> float:
        """
        Compute KL divergence between two Gaussian distributions.

        KL(P||Q) measures the information lost when using Q to approximate P.
        For Gaussians, this has a closed-form solution.

        Args:
            p_mean: Mean of distribution P (online/production)
            p_std: Standard deviation of P
            q_mean: Mean of distribution Q (baseline/training)
            q_std: Standard deviation of Q

        Returns:
            KL divergence value (non-negative)

        Formula:
            KL(P||Q) = log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²) / (2σ_Q²) - 0.5

        Note:
            KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P)
            By convention, we compute KL(online||baseline) to measure
            how much the online distribution differs from baseline.
        """
        eps = DriftMetrics.EPSILON

        # Ensure positive std
        p_std = max(p_std, eps)
        q_std = max(q_std, eps)

        # KL divergence formula for Gaussians
        log_term = math.log(q_std / p_std)
        variance_term = (p_std ** 2 + (p_mean - q_mean) ** 2) / (2 * q_std ** 2)

        kl = log_term + variance_term - 0.5

        # KL should be non-negative (numerical issues can cause small negatives)
        return max(0.0, kl)

    @staticmethod
    def symmetric_kl_divergence(
        p_mean: float,
        p_std: float,
        q_mean: float,
        q_std: float,
    ) -> float:
        """
        Compute symmetric KL divergence (Jensen-Shannon like).

        Average of KL(P||Q) and KL(Q||P) for a symmetric measure.

        Returns:
            (KL(P||Q) + KL(Q||P)) / 2
        """
        kl_pq = DriftMetrics.kl_divergence_gaussian(p_mean, p_std, q_mean, q_std)
        kl_qp = DriftMetrics.kl_divergence_gaussian(q_mean, q_std, p_mean, p_std)
        return (kl_pq + kl_qp) / 2

    @staticmethod
    def psi(
        baseline_proportions: List[float],
        online_proportions: List[float],
    ) -> float:
        """
        Compute Population Stability Index (PSI).

        PSI compares two distributions by binning and measuring the
        difference in proportions per bin.

        Args:
            baseline_proportions: Proportion of values in each bin (training)
            online_proportions: Proportion of values in each bin (production)

        Returns:
            PSI value (non-negative)

        Formula:
            PSI = Σ (online_% - baseline_%) × ln(online_% / baseline_%)

        Interpretation:
            PSI < 0.1: No significant change
            0.1 ≤ PSI < 0.2: Moderate change
            PSI ≥ 0.2: Significant change

        Note:
            Both lists must have the same length (same bins).
            Zero proportions are replaced with epsilon to avoid log(0).
        """
        if len(baseline_proportions) != len(online_proportions):
            raise ValueError("Proportions must have same length")

        if len(baseline_proportions) == 0:
            return 0.0

        eps = DriftMetrics.EPSILON
        psi_value = 0.0

        for baseline_p, online_p in zip(baseline_proportions, online_proportions):
            # Replace zeros with small epsilon
            baseline_p = max(baseline_p, eps)
            online_p = max(online_p, eps)

            # PSI term for this bin
            psi_value += (online_p - baseline_p) * math.log(online_p / baseline_p)

        return psi_value

    @classmethod
    def compute_severity(
        cls,
        mean_shift_z: float,
        kl_divergence: Optional[float] = None,
        psi: Optional[float] = None,
    ) -> DriftSeverity:
        """
        Determine overall drift severity from individual metrics.

        Uses the maximum severity across all available metrics.

        Args:
            mean_shift_z: Z-score of mean shift
            kl_divergence: KL divergence value (optional)
            psi: PSI value (optional)

        Returns:
            DriftSeverity enum value
        """
        severity = DriftSeverity.OK

        # Check mean shift
        if mean_shift_z >= cls.MEAN_SHIFT_ALERT_THRESHOLD:
            severity = DriftSeverity.ALERT
        elif mean_shift_z >= cls.MEAN_SHIFT_WARNING_THRESHOLD:
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # Check KL divergence
        if kl_divergence is not None:
            if kl_divergence >= cls.KL_ALERT_THRESHOLD:
                severity = DriftSeverity.ALERT
            elif kl_divergence >= cls.KL_WARNING_THRESHOLD:
                severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # Check PSI
        if psi is not None:
            if psi >= cls.PSI_ALERT_THRESHOLD:
                severity = DriftSeverity.ALERT
            elif psi >= cls.PSI_WARNING_THRESHOLD:
                severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        return severity

    @classmethod
    def compute_all(
        cls,
        feature_name: str,
        baseline_mean: float,
        baseline_std: float,
        online_mean: float,
        online_std: float,
        baseline_proportions: Optional[List[float]] = None,
        online_proportions: Optional[List[float]] = None,
    ) -> DriftResult:
        """
        Compute all drift metrics for a feature.

        Args:
            feature_name: Name of the feature
            baseline_mean: Mean from training data
            baseline_std: Standard deviation from training data
            online_mean: Mean from production data
            online_std: Standard deviation from production data
            baseline_proportions: Optional histogram proportions (training)
            online_proportions: Optional histogram proportions (production)

        Returns:
            DriftResult with all computed metrics
        """
        # Compute mean shift
        mean_shift_z = cls.mean_shift_zscore(baseline_mean, baseline_std, online_mean)

        # Compute KL divergence
        kl_div = cls.kl_divergence_gaussian(
            online_mean, online_std,
            baseline_mean, baseline_std
        )

        # Compute PSI if proportions are provided
        psi_value = None
        if baseline_proportions and online_proportions:
            try:
                psi_value = cls.psi(baseline_proportions, online_proportions)
            except ValueError as e:
                logger.warning(f"Could not compute PSI for {feature_name}: {e}")

        # Determine severity
        severity = cls.compute_severity(mean_shift_z, kl_div, psi_value)

        # Build details
        details = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "online_mean": online_mean,
            "online_std": online_std,
            "mean_diff": online_mean - baseline_mean,
            "std_ratio": online_std / baseline_std if baseline_std > 0 else None,
        }

        return DriftResult(
            feature_name=feature_name,
            mean_shift_z=mean_shift_z,
            kl_divergence=kl_div,
            psi=psi_value,
            severity=severity,
            details=details,
        )
