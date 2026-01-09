"""
Ranking model training package.

This package contains modules for training an XGBoost ranking model with
staged hyperparameter tuning and MLflow experiment tracking.

Modules:
    config: Configuration dataclasses
    metrics: Evaluation metrics (NDCG, RMSE, MAE)
    diagnostics: Train/val gap analysis
    trainer: Main RankingModelTrainer class

CLI Usage:
    python train_ranking_model.py
    python train_ranking_model.py --retune
    python train_ranking_model.py --use-existing
"""

from .config import DiagnosticResult, TrainingConfig, TuningResult
from .diagnostics import DiagnosticsAnalyzer
from .metrics import compute_mae, compute_ndcg_at_k, compute_rmse, evaluate_model
from .trainer import RankingModelTrainer

__all__ = [
    "TrainingConfig",
    "TuningResult",
    "DiagnosticResult",
    "DiagnosticsAnalyzer",
    "RankingModelTrainer",
    "compute_ndcg_at_k",
    "compute_rmse",
    "compute_mae",
    "evaluate_model",
]
