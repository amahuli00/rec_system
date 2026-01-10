"""
CLI entry point for data preparation.

This script prepares data for Two-Tower model training:
1. Loads train/val/test splits
2. Filters to positive interactions (rating >= threshold)
3. Builds ID mappings (user_id, movie_id → contiguous indices)
4. Saves processed data and mappings

Usage:
    python -m candidate_gen.data.prepare_data
    python -m candidate_gen.data.prepare_data --data-dir data/splits --output-dir candidate_gen/artifacts/data
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import DataConfig
from .dataset import IDMapper, build_user_positive_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_data(config: DataConfig) -> None:
    """
    Execute the complete data preparation pipeline.

    Steps:
    1. Load train/val/test splits from parquet files
    2. Filter to positive interactions (rating >= threshold)
    3. Build ID mappings from ALL training data
    4. Build user → positive items lookup for evaluation
    5. Save all outputs

    Args:
        config: Data configuration
    """
    logger.info("=" * 60)
    logger.info("DATA PREPARATION FOR TWO-TOWER MODEL")
    logger.info("=" * 60)

    # =========================================================================
    # Stage 1: Load data
    # =========================================================================
    logger.info("Loading data...")

    train_ratings = pd.read_parquet(config.data_dir / "train_ratings.parquet")
    val_ratings = pd.read_parquet(config.data_dir / "val_ratings.parquet")
    test_ratings = pd.read_parquet(config.data_dir / "test_ratings.parquet")

    logger.info(f"Train: {len(train_ratings):,} ratings")
    logger.info(f"Val:   {len(val_ratings):,} ratings")
    logger.info(f"Test:  {len(test_ratings):,} ratings")

    # =========================================================================
    # Stage 2: Filter to positive interactions
    # =========================================================================
    logger.info(f"Filtering to positive interactions (rating >= {config.positive_threshold})...")

    train_positive = train_ratings[train_ratings["rating"] >= config.positive_threshold].copy()
    val_positive = val_ratings[val_ratings["rating"] >= config.positive_threshold].copy()
    test_positive = test_ratings[test_ratings["rating"] >= config.positive_threshold].copy()

    logger.info(
        f"Train positive: {len(train_positive):,} "
        f"({len(train_positive)/len(train_ratings)*100:.1f}%)"
    )
    logger.info(
        f"Val positive:   {len(val_positive):,} "
        f"({len(val_positive)/len(val_ratings)*100:.1f}%)"
    )
    logger.info(
        f"Test positive:  {len(test_positive):,} "
        f"({len(test_positive)/len(test_ratings)*100:.1f}%)"
    )

    # =========================================================================
    # Stage 3: Build ID mappings
    # =========================================================================
    logger.info("Building ID mappings from ALL training data...")

    # Use ALL train ratings (not just positive) to include all users/items
    # This ensures we have embeddings for all entities seen in training
    id_mapper = IDMapper.from_dataframe(train_ratings)
    id_mapper.verify()

    logger.info(f"Number of users: {id_mapper.num_users:,}")
    logger.info(f"Number of items: {id_mapper.num_items:,}")
    logger.info("ID mappings verified: bijective and contiguous")

    # =========================================================================
    # Stage 4: Build user positive items for evaluation
    # =========================================================================
    logger.info("Building user positive items lookup...")

    train_user_positives = build_user_positive_items(train_positive, id_mapper)
    val_user_positives = build_user_positive_items(val_positive, id_mapper)

    logger.info(f"Users with train positives: {len(train_user_positives):,}")
    logger.info(f"Users with val positives:   {len(val_user_positives):,}")

    # =========================================================================
    # Stage 5: Save outputs
    # =========================================================================
    logger.info("Saving outputs...")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save positive interactions as parquet
    train_positive.to_parquet(output_dir / "train_positives.parquet", index=False)
    val_positive.to_parquet(output_dir / "val_positives.parquet", index=False)
    test_positive.to_parquet(output_dir / "test_positives.parquet", index=False)
    logger.info("Saved positive interaction parquets")

    # Save ID mappings
    id_mapper.save(output_dir)

    # Save user positive items as JSON (for evaluation)
    # Convert sets to lists and int keys to strings for JSON
    train_positives_json = {str(k): list(v) for k, v in train_user_positives.items()}
    val_positives_json = {str(k): list(v) for k, v in val_user_positives.items()}

    with open(output_dir / "train_user_positives.json", "w") as f:
        json.dump(train_positives_json, f)

    with open(output_dir / "val_user_positives.json", "w") as f:
        json.dump(val_positives_json, f)

    logger.info("Saved user positive items")

    # Save metadata
    metadata = {
        "positive_threshold": config.positive_threshold,
        "num_users": id_mapper.num_users,
        "num_items": id_mapper.num_items,
        "train_positives": len(train_positive),
        "val_positives": len(val_positive),
        "test_positives": len(test_positive),
        "users_with_train_positives": len(train_user_positives),
        "users_with_val_positives": len(val_user_positives),
        "data_dir": str(config.data_dir),
        "created_at": pd.Timestamp.now().isoformat(),
    }

    with open(output_dir / "data_config.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Files created:")
    for file in sorted(output_dir.glob("*")):
        size_kb = file.stat().st_size / 1024
        logger.info(f"  {file.name:30} ({size_kb:,.1f} KB)")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare data for Two-Tower candidate generation model"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train/val/test parquet splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("candidate_gen/artifacts/data"),
        help="Directory to save processed data and mappings",
    )
    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=4,
        help="Minimum rating to consider as positive interaction (default: 4)",
    )

    args = parser.parse_args()

    config = DataConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        positive_threshold=args.positive_threshold,
    )

    prepare_data(config)


if __name__ == "__main__":
    main()
