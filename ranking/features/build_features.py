"""
CLI entry point for feature building.

This script materializes features from the train/val/test splits for the
GBDT ranking model. Features are computed only from training data to
prevent data leakage.

Usage:
    python build_features.py                    # Run from ranking/features/
    python -m ranking.features.build_features   # Run as module from project root
    python -m ranking.features.build_features --save-serving-data  # Also save serving data

Output:
    - train_features.parquet
    - val_features.parquet
    - test_features.parquet
    - feature_metadata.json
    - README.md
    - user_stats.parquet (with --save-serving-data)
    - movie_stats.parquet (with --save-serving-data)
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build features for the GBDT ranking model."
    )
    parser.add_argument(
        "--save-serving-data",
        action="store_true",
        help="Save user_stats.parquet and movie_stats.parquet for serving",
    )
    parser.add_argument(
        "--serving-data-only",
        action="store_true",
        help="Only generate serving data (skip full feature build)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for feature building."""
    setup_logging()
    args = parse_args()

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

    if args.serving_data_only:
        # Only load data and compute aggregations, then save serving data
        builder._load_data()
        builder._compute_aggregation_features()
        builder.save_serving_data()
        print("\nServing data generated successfully")
    else:
        # Full feature build
        builder.run()

        # Optionally save serving data
        if args.save_serving_data:
            builder.save_serving_data()

        print("\nNext step: Train GBDT ranking model using these features")
        print("  python -m ranking.training.train_ranking_model")


if __name__ == "__main__":
    main()
