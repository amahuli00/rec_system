#!/usr/bin/env python
"""
CLI entry point for ranking model training.

Usage:
    # Default: auto-detect (use existing params if available, otherwise tune)
    python train_ranking_model.py

    # Force hyperparameter tuning (re-tune even if model exists)
    python train_ranking_model.py --retune

    # Use existing hyperparameters (skip tuning, faster training)
    python train_ranking_model.py --use-existing

    # Custom paths
    python train_ranking_model.py --features-dir ../features --models-dir ../models
"""

import argparse
import logging
import sys
from pathlib import Path

from config import TrainingConfig
from trainer import RankingModelTrainer


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the training script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost ranking model with optional hyperparameter tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default behavior (auto-detect)
  python train_ranking_model.py

  # Force re-tuning
  python train_ranking_model.py --retune

  # Skip tuning, use existing params
  python train_ranking_model.py --use-existing

  # Verbose output
  python train_ranking_model.py -v
        """,
    )

    # Path arguments
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("../features"),
        help="Directory containing feature parquet files (default: ../features)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("../models"),
        help="Directory to save trained models (default: ../models)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ranking_model",
        help="MLflow experiment name (default: ranking_model)",
    )

    # Tuning mode arguments (mutually exclusive)
    tuning_group = parser.add_mutually_exclusive_group()
    tuning_group.add_argument(
        "--retune",
        action="store_true",
        help="Force re-run hyperparameter tuning even if model exists",
    )
    tuning_group.add_argument(
        "--use-existing",
        action="store_true",
        help="Use hyperparameters from existing model_info.json (skip tuning)",
    )

    # Verbose output
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Validate paths
    if not args.features_dir.exists():
        logger.error(f"Features directory not found: {args.features_dir}")
        return 1

    # Create config
    config = TrainingConfig(
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        mlflow_experiment=args.experiment,
    )

    # Create trainer
    trainer = RankingModelTrainer(config)

    # Determine tuning mode
    if args.retune:
        skip_tuning = False
        logger.info("Mode: Force hyperparameter tuning")
    elif args.use_existing:
        skip_tuning = True
        logger.info("Mode: Use existing hyperparameters")
    else:
        skip_tuning = "auto"
        logger.info("Mode: Auto-detect (use existing if available)")

    # Run training
    try:
        model = trainer.run(skip_tuning=skip_tuning)
        logger.info("Training completed successfully!")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
