"""
Two-Tower model trainer with MLflow experiment tracking.

Implements:
- Training loop with in-batch negatives
- Validation with Recall@K
- Early stopping
- MLflow logging
- Model checkpointing
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import IDMapper, TwoTowerDataset
from ..model import ModelConfig, TwoTowerModel
from .config import TrainingConfig
from .losses import in_batch_softmax_loss
from .metrics import compute_baselines, compute_recall_at_k

logger = logging.getLogger(__name__)


class TwoTowerTrainer:
    """
    Trainer for Two-Tower candidate generation model.

    Implements the complete training pipeline:
    1. Load processed data and create DataLoader
    2. Initialize model and optimizer
    3. Training loop with in-batch negatives
    4. Validation with Recall@K
    5. Early stopping based on validation metric
    6. MLflow experiment tracking
    7. Save best model checkpoint

    Example:
        config = TrainingConfig(
            data_dir=Path("candidate_gen/artifacts/data"),
            output_dir=Path("candidate_gen/artifacts/models"),
            num_epochs=30,
        )
        trainer = TwoTowerTrainer(config)
        model, metrics = trainer.run()
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Will be populated during run()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.id_mapper: Optional[IDMapper] = None
        self.train_user_positives: Optional[Dict[int, Set[int]]] = None
        self.val_user_positives: Optional[Dict[int, Set[int]]] = None
        self.train_positive_df: Optional[pd.DataFrame] = None

        # Training state
        self.best_val_metric: float = 0.0
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0
        self.train_losses: list = []
        self.val_metrics_history: list = []

    def run(self) -> Tuple[nn.Module, Dict]:
        """
        Execute the complete training pipeline.

        Returns:
            Tuple of (trained_model, final_metrics)
        """
        logger.info("=" * 60)
        logger.info("TWO-TOWER MODEL TRAINING")
        logger.info("=" * 60)

        # Set random seeds
        self._set_seeds()

        # Stage 1: Load data
        self._load_data()

        # Stage 2: Create model
        self._create_model()

        # Stage 3: Setup MLflow
        self._setup_mlflow()

        # Stage 4: Training loop
        final_metrics = self._train()

        # Stage 5: Save model
        self._save_model()

        # Stage 6: Log final results
        self._log_final_results(final_metrics)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)

        return self.model, final_metrics

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        logger.info(f"Random seed: {self.config.seed}")

    def _load_data(self) -> None:
        """Load processed data and create DataLoader."""
        logger.info("Loading data...")

        data_dir = self.config.data_dir

        # Load ID mappings
        self.id_mapper = IDMapper.load(data_dir)
        logger.info(f"Users: {self.id_mapper.num_users:,}")
        logger.info(f"Items: {self.id_mapper.num_items:,}")

        # Load positive interactions
        self.train_positive_df = pd.read_parquet(data_dir / "train_positives.parquet")
        logger.info(f"Train positives: {len(self.train_positive_df):,}")

        # Load user positive items for evaluation
        with open(data_dir / "train_user_positives.json") as f:
            train_positives_json = json.load(f)
        self.train_user_positives = {
            int(k): set(v) for k, v in train_positives_json.items()
        }

        with open(data_dir / "val_user_positives.json") as f:
            val_positives_json = json.load(f)
        self.val_user_positives = {int(k): set(v) for k, v in val_positives_json.items()}
        logger.info(f"Val users: {len(self.val_user_positives):,}")

        # Create dataset and loader
        train_dataset = TwoTowerDataset(
            self.train_positive_df, self.id_mapper, verbose=True
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.info(f"Training batches: {len(self.train_loader):,}")

    def _create_model(self) -> None:
        """Initialize model and optimizer."""
        logger.info("Creating model...")

        model_config = ModelConfig(
            embedding_dim=self.config.embedding_dim,
            use_mlp=self.config.use_mlp,
        )

        self.model = TwoTowerModel(
            num_users=self.id_mapper.num_users,
            num_items=self.id_mapper.num_items,
            config=model_config,
        ).to(self.config.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        logger.info(f"Optimizer: Adam (lr={self.config.learning_rate})")

    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        # Start run
        run_name = self.config.run_name or f"dim{self.config.embedding_dim}_lr{self.config.learning_rate}"
        mlflow.start_run(run_name=run_name)

        # Log config
        mlflow.log_params(self.config.to_dict())
        logger.info(f"MLflow run started: {run_name}")

    def _train(self) -> Dict:
        """Execute training loop with early stopping."""
        logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            # Train one epoch
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Evaluate periodically
            if (epoch + 1) % self.config.eval_every_n_epochs == 0 or epoch == 0:
                val_metrics = self._validate()
                self.val_metrics_history.append(val_metrics)

                # Log metrics (replace @ with _at_ for MLflow compatibility)
                for metric, value in val_metrics.items():
                    mlflow_metric = metric.replace("@", "_at_")
                    mlflow.log_metric(f"val_{mlflow_metric}", value, step=epoch)

                # Print metrics
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                logger.info(f"Epoch {epoch + 1}: {metrics_str}")

                # Early stopping check
                primary_metric = val_metrics.get("recall@50", 0.0)
                if primary_metric > self.best_val_metric + self.config.min_delta:
                    self.best_val_metric = primary_metric
                    self.best_epoch = epoch + 1
                    self.epochs_without_improvement = 0
                    self._save_checkpoint("best")
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {self.config.patience} evals)"
                    )
                    break

        # Load best model
        self._load_checkpoint("best")

        # Final evaluation
        final_metrics = self._validate()

        return final_metrics

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
        )
        for batch in pbar:
            user_ids = batch["user_idx"].to(self.config.device)
            pos_items = batch["pos_item_idx"].to(self.config.device)

            self.optimizer.zero_grad()

            user_emb, item_emb = self.model(user_ids, pos_items)
            loss = in_batch_softmax_loss(user_emb, item_emb, self.config.temperature)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def _validate(self) -> Dict[str, float]:
        """Compute validation metrics."""
        return compute_recall_at_k(
            model=self.model,
            user_positives=self.val_user_positives,
            num_items=self.id_mapper.num_items,
            k_values=self.config.eval_k_values,
            device=self.config.device,
            show_progress=False,
        )

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "best_val_metric": self.best_val_metric,
                "best_epoch": self.best_epoch,
            },
            checkpoint_path,
        )

    def _load_checkpoint(self, name: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = self.config.output_dir / "checkpoints" / f"{name}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint: {name}")

    def _save_model(self) -> None:
        """Save final model and metadata."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / "two_tower_model.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved: {model_path}")

        # Save model config
        model_info = {
            "num_users": self.id_mapper.num_users,
            "num_items": self.id_mapper.num_items,
            "embedding_dim": self.config.embedding_dim,
            "use_mlp": self.config.use_mlp,
            "training_config": self.config.to_dict(),
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        with open(output_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        # Log to MLflow
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(output_dir / "model_info.json"))

    def _log_final_results(self, final_metrics: Dict) -> None:
        """Log final results to MLflow and console."""
        # Compute baselines for comparison
        baselines = compute_baselines(
            user_positives=self.val_user_positives,
            train_positive_df=self.train_positive_df,
            item_to_idx=self.id_mapper.item_to_idx,
            num_items=self.id_mapper.num_items,
            k_values=self.config.eval_k_values,
        )

        # Log baselines (replace @ with _at_ for MLflow compatibility)
        for baseline_name, baseline_metrics in baselines.items():
            for metric, value in baseline_metrics.items():
                mlflow_metric = metric.replace("@", "_at_")
                mlflow.log_metric(f"{baseline_name}_{mlflow_metric}", value)

        # Print comparison
        logger.info("\nFinal Results:")
        logger.info("-" * 40)

        for k in self.config.eval_k_values:
            metric_key = f"recall@{k}"
            model_val = final_metrics.get(metric_key, 0.0)
            pop_val = baselines["popularity"].get(metric_key, 0.0)
            rand_val = baselines["random"].get(metric_key, 0.0)

            logger.info(f"  {metric_key}:")
            logger.info(f"    Model:      {model_val:.4f}")
            logger.info(f"    Popularity: {pop_val:.4f}")
            logger.info(f"    Random:     {rand_val:.4f}")
            logger.info(f"    Lift vs Pop: {(model_val / pop_val - 1) * 100:.1f}%")

        mlflow.end_run()
