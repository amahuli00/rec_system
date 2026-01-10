"""
Configuration dataclasses for data preparation.

This module defines configuration for converting MovieLens ratings
to implicit feedback format for Two-Tower model training.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """
    Configuration for data preparation.

    Attributes:
        data_dir: Directory containing train/val/test parquet splits
        output_dir: Directory to save processed data and mappings
        positive_threshold: Minimum rating to consider as positive (default: 4)
            Ratings >= this value are treated as positive interactions.
            Ratings below are excluded from training.
    """

    data_dir: Path = field(default_factory=lambda: Path("data/splits"))
    output_dir: Path = field(default_factory=lambda: Path("candidate_gen/artifacts/data"))
    positive_threshold: int = 4

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
