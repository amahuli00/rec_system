"""
Diagnostic analysis for train/val gap detection.

This module provides the DiagnosticsAnalyzer class that evaluates model performance
on both training and validation sets to detect overfitting or underfitting.

Decision Rules:
- NDCG gap > threshold (0.05): Overfitting -> REDUCE_COMPLEXITY
- NDCG gap < 0.02 AND val_ndcg < 0.70: Underfitting -> INCREASE_COMPLEXITY
- Otherwise: Good fit -> No action needed
"""

import logging
from typing import Tuple

import pandas as pd

try:
    from .config import DiagnosticResult, TrainingConfig
    from .metrics import evaluate_model
except ImportError:
    from config import DiagnosticResult, TrainingConfig
    from metrics import evaluate_model

logger = logging.getLogger(__name__)


class DiagnosticsAnalyzer:
    """
    Analyzes train/val gap to determine if hyperparameter tuning is needed.

    This class implements Stage 2 of the training pipeline, which evaluates
    the best model from Stage 1 to check for overfitting or underfitting.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the diagnostics analyzer.

        Args:
            config: Training configuration with threshold settings
        """
        self.config = config

    def analyze(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_df: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_df: pd.DataFrame,
    ) -> DiagnosticResult:
        """
        Analyze model performance to detect overfitting/underfitting.

        Args:
            model: Trained XGBoost model
            X_train: Training features
            y_train: Training targets
            train_df: Training DataFrame (for NDCG computation)
            X_val: Validation features
            y_val: Validation targets
            val_df: Validation DataFrame (for NDCG computation)

        Returns:
            DiagnosticResult with gap analysis and recommendations
        """
        logger.info("Running diagnostic analysis (train vs val gap)...")

        # Evaluate on both train and val
        train_metrics, _ = evaluate_model(model, X_train, y_train, train_df, prefix="train_")
        val_metrics, _ = evaluate_model(model, X_val, y_val, val_df, prefix="val_")

        # Compute gaps
        ndcg_gap = train_metrics["train_ndcg_10"] - val_metrics["val_ndcg_10"]
        rmse_gap = val_metrics["val_rmse"] - train_metrics["train_rmse"]

        # Log results
        logger.info(f"Train NDCG@10: {train_metrics['train_ndcg_10']:.4f}")
        logger.info(f"Val NDCG@10:   {val_metrics['val_ndcg_10']:.4f}")
        logger.info(f"NDCG Gap:      {ndcg_gap:.4f}")
        logger.info(f"Train RMSE:    {train_metrics['train_rmse']:.4f}")
        logger.info(f"Val RMSE:      {val_metrics['val_rmse']:.4f}")
        logger.info(f"RMSE Gap:      {rmse_gap:.4f}")

        # Determine action
        proceed_to_stage3, stage3_action = self._determine_action(
            ndcg_gap=ndcg_gap,
            val_ndcg_10=val_metrics["val_ndcg_10"],
        )

        result = DiagnosticResult(
            train_ndcg_10=train_metrics["train_ndcg_10"],
            val_ndcg_10=val_metrics["val_ndcg_10"],
            ndcg_gap=ndcg_gap,
            train_rmse=train_metrics["train_rmse"],
            val_rmse=val_metrics["val_rmse"],
            rmse_gap=rmse_gap,
            proceed_to_stage3=proceed_to_stage3,
            stage3_action=stage3_action,
        )

        self._log_decision(result)
        return result

    def _determine_action(
        self,
        ndcg_gap: float,
        val_ndcg_10: float,
    ) -> Tuple[bool, str]:
        """
        Determine whether to proceed to Stage 3 and what action to take.

        Args:
            ndcg_gap: Gap between train and val NDCG@10
            val_ndcg_10: Validation NDCG@10

        Returns:
            Tuple of (proceed_to_stage3, stage3_action)
        """
        # Overfitting: Large gap between train and val
        if ndcg_gap > self.config.ndcg_gap_threshold:
            return True, "REDUCE_COMPLEXITY"

        # Underfitting: Small gap but poor performance
        if ndcg_gap < 0.02 and val_ndcg_10 < self.config.underfitting_ndcg_threshold:
            return True, "INCREASE_COMPLEXITY"

        # Good fit: No action needed
        return False, None

    def _log_decision(self, result: DiagnosticResult) -> None:
        """Log the diagnostic decision."""
        if result.stage3_action == "REDUCE_COMPLEXITY":
            logger.warning(
                f"OVERFITTING detected: NDCG gap ({result.ndcg_gap:.4f}) > "
                f"threshold ({self.config.ndcg_gap_threshold})"
            )
            logger.info("Recommendation: Reduce complexity (lower max_depth or colsample_bytree)")
        elif result.stage3_action == "INCREASE_COMPLEXITY":
            logger.warning(
                f"UNDERFITTING detected: Small gap but low NDCG@10 ({result.val_ndcg_10:.4f})"
            )
            logger.info("Recommendation: Increase complexity (higher max_depth)")
        else:
            logger.info(
                f"GOOD FIT detected: NDCG gap ({result.ndcg_gap:.4f}) is acceptable, "
                f"Val NDCG@10 ({result.val_ndcg_10:.4f}) is satisfactory"
            )
            logger.info("No further tuning needed - skipping Stage 3")
