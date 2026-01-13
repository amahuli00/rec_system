"""
Drift Detection Package

Provides feature drift detection capabilities for the recommendation service.
Drift detection monitors whether the distribution of features in production
matches the distribution seen during training.

Components:
- BaselineStatistics: Computes and stores training data statistics
- OnlineStatistics: Accumulates streaming statistics using Welford's algorithm
- DriftMetrics: Computes KL divergence, PSI, and mean shift
- DriftDetector: Compares baseline vs online stats and generates alerts

Why Drift Detection Matters:
============================

ML models are trained on historical data and assume production data will
follow similar patterns. When production data "drifts" from training data,
model predictions may become unreliable.

Common causes of drift:
1. User behavior changes (seasonality, trends)
2. Data pipeline bugs (nulls, encoding changes)
3. Feature engineering changes
4. Population changes (new users with different behavior)

Types of drift we detect:
1. Mean shift: Average feature value changed significantly
2. Variance shift: Spread of values changed
3. Distribution shift: Overall shape of distribution changed (KL divergence)

Alert thresholds:
- Mean shift > 3 std: Warning
- PSI > 0.1: Moderate change
- PSI > 0.2: Significant change requiring investigation
"""

from .baseline import BaselineStatistics
from .online_stats import OnlineStatistics
from .metrics import DriftMetrics
from .detector import DriftDetector

__all__ = [
    "BaselineStatistics",
    "OnlineStatistics",
    "DriftMetrics",
    "DriftDetector",
]
