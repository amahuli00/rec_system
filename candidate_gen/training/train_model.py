"""
CLI entry point for Two-Tower model training.

Usage:
    # Train with default config
    python -m candidate_gen.training.train_model

    # Train with custom hyperparameters
    python -m candidate_gen.training.train_model --embedding-dim 256 --lr 0.0001

    # Train with YAML config
    python -m candidate_gen.training.train_model --config experiments/baseline.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml

from .config import TrainingConfig
from .trainer import TwoTowerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config_from_yaml(yaml_path: Path) -> TrainingConfig:
    """Load training config from YAML file."""
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    # Handle nested structure if present (for experiment configs)
    if "base_config" in config_dict:
        config_dict = config_dict["base_config"]

    # Convert paths
    if "data_dir" in config_dict:
        config_dict["data_dir"] = Path(config_dict["data_dir"])
    if "output_dir" in config_dict:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    return TrainingConfig(**config_dict)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Two-Tower candidate generation model"
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file",
    )

    # Model hyperparameters
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--use-mlp",
        action="store_true",
        help="Use MLP layers in towers",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 1024)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (default: 1e-6)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 30)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature (default: 0.1)",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 5)",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: candidate_gen/artifacts/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: candidate_gen/artifacts/models)",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )

    args = parser.parse_args()

    # Load base config
    if args.config:
        config = load_config_from_yaml(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = TrainingConfig()

    # Override with CLI arguments
    if args.embedding_dim is not None:
        config.embedding_dim = args.embedding_dim
    if args.use_mlp:
        config.use_mlp = True
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.patience is not None:
        config.patience = args.patience
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.run_name is not None:
        config.run_name = args.run_name

    # Train
    trainer = TwoTowerTrainer(config)
    model, metrics = trainer.run()

    # Print final metrics
    logger.info("\nFinal Validation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
