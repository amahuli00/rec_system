"""
Configuration for feature building pipeline.

This module defines configuration dataclasses for the feature engineering
process, following the same pattern as training/config.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FeatureConfig:
    """
    Configuration for the feature building pipeline.

    Attributes:
        data_dir: Directory containing train/val/test splits and metadata.
        output_dir: Directory to save feature parquets and metadata.
        compute_interaction_features: Whether to compute user-movie interaction features.
        fill_missing_with_zero: Whether to fill remaining NaN values with 0.
    """

    data_dir: Path = field(default_factory=lambda: Path("../../data/splits"))
    output_dir: Path = field(default_factory=lambda: Path("."))
    compute_interaction_features: bool = True
    fill_missing_with_zero: bool = True

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class FeatureMetadata:
    """
    Metadata about generated features.

    This is populated during feature building and saved to JSON for
    downstream consumers (e.g., the ranking model trainer).

    Attributes:
        version: Feature set version string.
        created_at: ISO timestamp of feature generation.
        train_timestamp_max: Maximum timestamp in training data.
        num_features: Total number of feature columns.
        feature_groups: Mapping of group names to feature column lists.
        cold_start_defaults: Default values for cold-start users/movies.
        split_sizes: Number of rows in each split.
        genre_columns: List of genre feature column names.
    """

    version: str
    created_at: str
    train_timestamp_max: int
    num_features: int
    feature_groups: Dict[str, List[str]]
    cold_start_defaults: Dict[str, float]
    split_sizes: Dict[str, int]
    genre_columns: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "train_timestamp_max": self.train_timestamp_max,
            "num_features": self.num_features,
            "feature_groups": self.feature_groups,
            "cold_start_defaults": self.cold_start_defaults,
            "split_sizes": self.split_sizes,
        }
