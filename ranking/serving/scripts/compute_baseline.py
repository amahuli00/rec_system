#!/usr/bin/env python
"""
Compute baseline statistics for drift detection.

This script computes baseline statistics from training data and saves them
to a JSON file. These statistics serve as the reference for detecting
feature drift in production.

Usage:
    # From project root
    PYTHONPATH=$(pwd) python ranking/serving/scripts/compute_baseline.py

    # Verify output
    cat ranking/features/baseline_stats.json | head -50

The output file (baseline_stats.json) should be:
1. Versioned alongside model artifacts
2. Updated when retraining the model
3. Deployed to production with the model

Design Decisions:
1. Compute from training split only (not validation/test) to avoid leakage
2. Use all feature columns that will be used during inference
3. Store histogram bins for PSI computation
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.serving.drift import BaselineStatistics
from ranking.shared_utils import FEATURES_DIR, load_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_features() -> pd.DataFrame:
    """
    Load training features from parquet file.

    Returns:
        DataFrame with training features
    """
    train_path = FEATURES_DIR / "train_features.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found at {train_path}")

    df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(df)} training samples")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline statistics for drift detection"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURES_DIR / "baseline_stats.json",
        help="Output path for baseline statistics (default: ranking/features/baseline_stats.json)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output after saving"
    )

    args = parser.parse_args()

    try:
        # Load training features
        df = load_training_features()

        # Get feature columns
        feature_columns = load_feature_columns()
        logger.info(f"Computing baseline for {len(feature_columns)} features")

        # Compute baseline statistics
        baseline = BaselineStatistics.from_dataframe(df, feature_columns)

        # Save to file
        baseline.save(args.output)
        logger.info(f"Saved baseline statistics to {args.output}")

        # Verify if requested
        if args.verify:
            loaded = BaselineStatistics.load(args.output)
            logger.info(f"Verified: {len(loaded.features)} features loaded")

            # Print sample statistics
            for name in list(loaded.features.keys())[:3]:
                feat = loaded.features[name]
                logger.info(
                    f"  {name}: mean={feat.mean:.4f}, std={feat.std:.4f}, "
                    f"n={feat.n_samples}"
                )

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to compute baseline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
