"""
Training module for Two-Tower candidate generation.

This module provides:
- TrainingConfig: Configuration for training
- Loss functions: in_batch_softmax_loss, bpr_loss
- Metrics: compute_recall_at_k
- TwoTowerTrainer: Complete training pipeline with MLflow
- ExperimentConfig: Configuration for hyperparameter sweeps
- ExperimentRunner: Run grid search experiments
"""

from .config import TrainingConfig
from .experiment_config import ExperimentConfig
from .experiment_runner import ExperimentRunner
from .losses import bpr_loss, in_batch_softmax_loss
from .metrics import compute_baselines, compute_recall_at_k
from .trainer import TwoTowerTrainer

__all__ = [
    "TrainingConfig",
    "ExperimentConfig",
    "ExperimentRunner",
    "in_batch_softmax_loss",
    "bpr_loss",
    "compute_recall_at_k",
    "compute_baselines",
    "TwoTowerTrainer",
]
