"""
Baseline statistics for drift detection.

This module computes and stores statistics from training data. These baseline
statistics serve as the reference point for detecting drift - we compare
online (production) statistics against these baseline values.

Design Decisions:

1. **Compute once, store permanently**: Baseline stats are computed from the
   training data and never change. This ensures we're comparing production
   data against the same reference used during model development.

2. **Store mean, std, min, max, and histogram bins**: This gives us multiple
   ways to detect drift:
   - Mean/std: For Z-score based drift detection
   - Min/max: For range violation detection
   - Histogram bins: For PSI (Population Stability Index)

3. **JSON serialization**: Stats are stored as JSON for easy inspection,
   versioning, and deployment alongside model artifacts.

4. **Per-feature granularity**: Each feature has its own baseline stats,
   allowing us to pinpoint which specific features are drifting.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureBaseline:
    """
    Baseline statistics for a single feature.

    These statistics are computed from training data and serve as the
    reference for drift detection.

    Attributes:
        name: Feature name
        mean: Mean value in training data
        std: Standard deviation in training data
        min: Minimum value in training data
        max: Maximum value in training data
        percentiles: Dictionary of percentile values (5, 25, 50, 75, 95)
        histogram_bins: Bin edges for PSI computation
        histogram_counts: Counts per bin (normalized to proportions)
        n_samples: Number of samples used to compute stats
    """
    name: str
    mean: float
    std: float
    min: float
    max: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[float] = field(default_factory=list)
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureBaseline":
        """Create from dictionary."""
        return cls(**data)


class BaselineStatistics:
    """
    Computes and stores baseline statistics from training data.

    This class provides methods to:
    1. Compute statistics from a training DataFrame
    2. Save/load statistics to/from JSON
    3. Access individual feature baselines for comparison

    Example:
        >>> # Compute from training data
        >>> baseline = BaselineStatistics.from_dataframe(train_df, feature_columns)
        >>> baseline.save("baselines.json")

        >>> # Load and use
        >>> baseline = BaselineStatistics.load("baselines.json")
        >>> user_rating_baseline = baseline.get_feature("user_avg_rating")
        >>> print(f"Training mean: {user_rating_baseline.mean}")
    """

    # Number of histogram bins for PSI computation
    N_BINS = 10

    def __init__(self):
        """Initialize empty baseline statistics."""
        self.features: Dict[str, FeatureBaseline] = {}
        self.computed_at: Optional[str] = None
        self.n_samples: int = 0

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> "BaselineStatistics":
        """
        Compute baseline statistics from a training DataFrame.

        Args:
            df: Training data DataFrame
            feature_columns: List of feature column names to compute stats for

        Returns:
            BaselineStatistics instance with computed values
        """
        instance = cls()
        instance.n_samples = len(df)
        instance.computed_at = pd.Timestamp.now().isoformat()

        logger.info(f"Computing baseline statistics for {len(feature_columns)} features")

        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Feature {col} not found in DataFrame, skipping")
                continue

            values = df[col].dropna()
            if len(values) == 0:
                logger.warning(f"Feature {col} has no valid values, skipping")
                continue

            # Compute basic statistics
            mean = float(values.mean())
            std = float(values.std())
            min_val = float(values.min())
            max_val = float(values.max())

            # Compute percentiles
            percentiles = {
                "5": float(np.percentile(values, 5)),
                "25": float(np.percentile(values, 25)),
                "50": float(np.percentile(values, 50)),
                "75": float(np.percentile(values, 75)),
                "95": float(np.percentile(values, 95)),
            }

            # Compute histogram for PSI
            # Use quantile-based bins for more robust comparison
            bin_edges = np.percentile(
                values,
                np.linspace(0, 100, cls.N_BINS + 1)
            ).tolist()

            # Make bin edges unique (handle constant features)
            bin_edges = sorted(set(bin_edges))
            if len(bin_edges) < 2:
                # Constant feature - use min-max range
                bin_edges = [min_val, max_val + 1e-10]

            # Compute histogram counts (as proportions)
            counts, _ = np.histogram(values, bins=bin_edges)
            proportions = (counts / counts.sum()).tolist()

            instance.features[col] = FeatureBaseline(
                name=col,
                mean=mean,
                std=std if std > 0 else 1e-10,  # Avoid division by zero
                min=min_val,
                max=max_val,
                percentiles=percentiles,
                histogram_bins=bin_edges,
                histogram_counts=proportions,
                n_samples=len(values),
            )

        logger.info(f"Computed baseline for {len(instance.features)} features")
        return instance

    def get_feature(self, name: str) -> Optional[FeatureBaseline]:
        """Get baseline statistics for a specific feature."""
        return self.features.get(name)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names with baselines."""
        return list(self.features.keys())

    def save(self, path: Path) -> None:
        """
        Save baseline statistics to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        data = {
            "computed_at": self.computed_at,
            "n_samples": self.n_samples,
            "features": {
                name: feat.to_dict()
                for name, feat in self.features.items()
            }
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved baseline statistics to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaselineStatistics":
        """
        Load baseline statistics from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            BaselineStatistics instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        instance = cls()
        instance.computed_at = data.get("computed_at")
        instance.n_samples = data.get("n_samples", 0)

        for name, feat_data in data.get("features", {}).items():
            instance.features[name] = FeatureBaseline.from_dict(feat_data)

        logger.info(f"Loaded baseline statistics from {path}")
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "computed_at": self.computed_at,
            "n_samples": self.n_samples,
            "features": {
                name: feat.to_dict()
                for name, feat in self.features.items()
            }
        }
