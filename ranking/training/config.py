"""
Configuration dataclasses for ranking model training.

This module contains all configuration and result dataclasses used by the
training pipeline. Using dataclasses provides type safety and clear documentation
of expected values.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for the ranking model training pipeline.

    Attributes:
        features_dir: Directory containing feature parquet files
        models_dir: Directory to save trained models
        mlflow_tracking_uri: MLflow tracking URI (file-based by default)
        mlflow_experiment: Name of the MLflow experiment
        learning_rates: Learning rates to try in Stage 1 tuning
        max_depth: Default tree depth
        colsample_bytree: Feature sampling ratio
        n_estimators_max: Maximum trees for early stopping
        early_stopping_rounds: Rounds without improvement to stop
        ndcg_gap_threshold: Threshold for overfitting detection
        underfitting_ndcg_threshold: NDCG below this with small gap = underfitting
        random_seed: Random seed for reproducibility
    """
    features_dir: Path = field(default_factory=lambda: Path("../features"))
    models_dir: Path = field(default_factory=lambda: Path("../models"))
    mlflow_tracking_uri: str = "file:///Users/ashishmahuli/Desktop/rec_system/mlruns"
    mlflow_experiment: str = "ranking_model"

    # Hyperparameter tuning settings
    learning_rates: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2])
    max_depth: int = 6
    colsample_bytree: float = 0.8
    n_estimators_max: int = 5000
    early_stopping_rounds: int = 50

    # Diagnostic thresholds
    ndcg_gap_threshold: float = 0.05  # Gap > this = overfitting
    underfitting_ndcg_threshold: float = 0.70  # NDCG < this with small gap = underfitting

    # Reproducibility
    random_seed: int = 42


@dataclass
class TuningResult:
    """
    Result from a single hyperparameter tuning run.

    Attributes:
        learning_rate: Learning rate used
        max_depth: Tree depth used
        best_iteration: Best iteration from early stopping
        val_ndcg_10: Validation NDCG@10
        val_ndcg_20: Validation NDCG@20
        val_rmse: Validation RMSE
        val_mae: Validation MAE
        model: Trained XGBoost model
    """
    learning_rate: float
    max_depth: int
    best_iteration: int
    val_ndcg_10: float
    val_ndcg_20: float
    val_rmse: float
    val_mae: float
    model: Any  # xgb.XGBRegressor


@dataclass
class DiagnosticResult:
    """
    Result from train/val gap diagnostic analysis.

    Attributes:
        train_ndcg_10: Training NDCG@10
        val_ndcg_10: Validation NDCG@10
        ndcg_gap: Gap between train and val NDCG@10
        train_rmse: Training RMSE
        val_rmse: Validation RMSE
        rmse_gap: Gap between val and train RMSE
        proceed_to_stage3: Whether to run Stage 3 tuning
        stage3_action: Action to take in Stage 3 (REDUCE_COMPLEXITY, INCREASE_COMPLEXITY, or None)
    """
    train_ndcg_10: float
    val_ndcg_10: float
    ndcg_gap: float
    train_rmse: float
    val_rmse: float
    rmse_gap: float
    proceed_to_stage3: bool
    stage3_action: Optional[str]  # "REDUCE_COMPLEXITY", "INCREASE_COMPLEXITY", or None
