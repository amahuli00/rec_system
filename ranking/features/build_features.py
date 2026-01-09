"""
CLI entry point for feature building.

This script materializes features from the train/val/test splits for the
GBDT ranking model. Features are computed only from training data to
prevent data leakage.

Usage:
    python build_features.py                    # Run from ranking/features/
    python -m ranking.features.build_features   # Run as module from project root

Output:
    - train_features.parquet
    - val_features.parquet
    - test_features.parquet
    - feature_metadata.json
    - README.md
"""

import logging
import sys
from pathlib import Path


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Main entry point for feature building."""
    setup_logging()

    # Import here to avoid circular imports when running as script
    from .builders import FeatureBuilder
    from .config import FeatureConfig

    # Set up paths relative to this file
    features_dir = Path(__file__).parent
    data_dir = features_dir.parent.parent / "data" / "splits"

    # Create config
    config = FeatureConfig(
        data_dir=data_dir,
        output_dir=features_dir,
    )

    # Build features
    builder = FeatureBuilder(config)
    builder.run()

    print("\nNext step: Train GBDT ranking model using these features")
    print("  python -m ranking.training.train_ranking_model")


if __name__ == "__main__":
    main()
