"""
Ranking model trainer with 3-stage hyperparameter tuning.

This module contains the RankingModelTrainer class that orchestrates the full
training pipeline:
- Stage 1: Primary tuning (learning rate grid search with early stopping)
- Stage 2: Diagnostics (train/val gap analysis)
- Stage 3: Secondary tuning (only if needed based on diagnostics)

The trainer also supports skipping tuning by using existing hyperparameters
from a previously trained model.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from .config import DiagnosticResult, TrainingConfig, TuningResult
    from .diagnostics import DiagnosticsAnalyzer
    from .metrics import evaluate_model
except ImportError:
    from config import DiagnosticResult, TrainingConfig, TuningResult
    from diagnostics import DiagnosticsAnalyzer
    from metrics import evaluate_model

logger = logging.getLogger(__name__)


class RankingModelTrainer:
    """
    Trains XGBoost ranking model with staged hyperparameter tuning.

    The trainer implements a production-style 3-stage tuning approach:
    1. Primary tuning: Grid search learning_rate with early stopping
    2. Diagnostics: Analyze train/val gap for overfitting/underfitting
    3. Secondary tuning: Adjust complexity only if diagnostics indicate need

    Supports both full tuning and using existing hyperparameters.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.diagnostics_analyzer = DiagnosticsAnalyzer(config)

        # Data attributes (populated by load_data)
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_cols: Optional[List[str]] = None
        self.metadata: Optional[dict] = None

        # Result attributes
        self.best_model: Optional[xgb.XGBRegressor] = None
        self.best_params: Optional[dict] = None
        self.diagnostic_result: Optional[DiagnosticResult] = None
        self.stage3_results: List[TuningResult] = []

    def run(self, skip_tuning: Union[bool, str] = "auto") -> xgb.XGBRegressor:
        """
        Execute the training pipeline.

        Args:
            skip_tuning: Controls hyperparameter tuning behavior
                - False: Always run full tuning (3-stage approach)
                - True: Load params from existing model_info.json, skip tuning
                - 'auto': Use existing if model_info.json exists, else tune

        Returns:
            Trained XGBoost model
        """
        # Configure MLflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        # Load data
        self.load_data()

        # Determine if we should skip tuning
        existing_params = self._load_existing_params()

        if skip_tuning == "auto":
            skip_tuning = existing_params is not None

        if skip_tuning and existing_params:
            logger.info(f"Using existing hyperparameters: {existing_params}")
            self.best_model = self._train_with_params(existing_params)
            self.best_params = existing_params
        else:
            if skip_tuning and existing_params is None:
                logger.warning("--use-existing specified but no existing params found. Running full tuning.")

            # Full tuning pipeline
            logger.info("Running full hyperparameter tuning pipeline...")

            # Stage 1: Primary tuning
            best_stage1 = self._run_stage1_primary_tuning()

            # Stage 2: Diagnostics
            self.diagnostic_result = self._run_stage2_diagnostics(best_stage1)

            # Stage 3: Secondary tuning (if needed)
            if self.diagnostic_result.proceed_to_stage3:
                self.best_model, self.best_params = self._run_stage3_secondary_tuning(best_stage1)
            else:
                self.best_model = best_stage1.model
                self.best_params = {
                    "learning_rate": best_stage1.learning_rate,
                    "max_depth": best_stage1.max_depth,
                    "n_estimators": best_stage1.best_iteration,
                    "colsample_bytree": self.config.colsample_bytree,
                }

        # Save model
        self._save_model()

        # Final evaluation
        self._final_evaluation()

        return self.best_model

    def load_data(self) -> None:
        """Load feature data from parquet files."""
        logger.info(f"Loading data from {self.config.features_dir}...")

        features_dir = self.config.features_dir

        self.train_df = pd.read_parquet(features_dir / "train_features.parquet")
        self.val_df = pd.read_parquet(features_dir / "val_features.parquet")
        self.test_df = pd.read_parquet(features_dir / "test_features.parquet")

        # Load metadata
        with open(features_dir / "feature_metadata.json", "r") as f:
            self.metadata = json.load(f)

        # Separate features from target
        exclude_cols = ["user_id", "movie_id", "rating", "timestamp"]
        self.feature_cols = [col for col in self.train_df.columns if col not in exclude_cols]

        self.X_train = self.train_df[self.feature_cols]
        self.y_train = self.train_df["rating"]
        self.X_val = self.val_df[self.feature_cols]
        self.y_val = self.val_df["rating"]
        self.X_test = self.test_df[self.feature_cols]
        self.y_test = self.test_df["rating"]

        logger.info(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        logger.info(f"Features: {len(self.feature_cols)}")

    def _load_existing_params(self) -> Optional[dict]:
        """
        Load hyperparameters from existing model_info.json.

        Returns:
            Dictionary of hyperparameters, or None if file doesn't exist
        """
        model_info_path = self.config.models_dir / "model_info.json"

        if not model_info_path.exists():
            return None

        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            return model_info.get("params", None)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load existing params: {e}")
            return None

    def _train_with_params(self, params: dict) -> xgb.XGBRegressor:
        """
        Train model with specified hyperparameters (no tuning).

        Args:
            params: Dictionary with learning_rate, max_depth, n_estimators, colsample_bytree

        Returns:
            Trained XGBoost model
        """
        logger.info("Training with existing hyperparameters (skipping tuning)...")

        with mlflow.start_run(run_name="train_with_existing_params"):
            mlflow.log_params(params)
            mlflow.log_param("mode", "skip_tuning")

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                colsample_bytree=params.get("colsample_bytree", self.config.colsample_bytree),
                random_state=self.config.random_seed,
                n_jobs=-1,
            )

            model.fit(self.X_train, self.y_train, verbose=False)

            # Evaluate
            val_metrics, _ = evaluate_model(model, self.X_val, self.y_val, self.val_df, prefix="val_")
            mlflow.log_metrics(val_metrics)

            logger.info(f"Val NDCG@10: {val_metrics['val_ndcg_10']:.4f}")
            logger.info(f"Val RMSE:    {val_metrics['val_rmse']:.4f}")

        return model

    def _run_stage1_primary_tuning(self) -> TuningResult:
        """
        Stage 1: Grid search over learning rates with early stopping.

        Returns:
            Best TuningResult from Stage 1
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: PRIMARY TUNING (Learning Rate + Early Stopping)")
        logger.info("=" * 60)

        results: List[TuningResult] = []

        for lr in self.config.learning_rates:
            logger.info(f"Training with learning_rate={lr}...")

            with mlflow.start_run(run_name=f"stage1_lr_{lr}"):
                params = {
                    "max_depth": self.config.max_depth,
                    "learning_rate": lr,
                    "n_estimators": self.config.n_estimators_max,
                    "colsample_bytree": self.config.colsample_bytree,
                    "early_stopping_rounds": self.config.early_stopping_rounds,
                    "objective": "reg:squarederror",
                    "random_state": self.config.random_seed,
                    "n_jobs": -1,
                }

                mlflow.log_params(params)
                mlflow.log_param("stage", "primary_tuning")

                model = xgb.XGBRegressor(**params)
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False,
                )

                # Evaluate
                val_metrics, _ = evaluate_model(model, self.X_val, self.y_val, self.val_df, prefix="val_")
                mlflow.log_metrics(val_metrics)
                mlflow.log_param("best_iteration", model.best_iteration)

                result = TuningResult(
                    learning_rate=lr,
                    max_depth=self.config.max_depth,
                    best_iteration=model.best_iteration,
                    val_ndcg_10=val_metrics["val_ndcg_10"],
                    val_ndcg_20=val_metrics["val_ndcg_20"],
                    val_rmse=val_metrics["val_rmse"],
                    val_mae=val_metrics["val_mae"],
                    model=model,
                )
                results.append(result)

                logger.info(f"  Best iteration: {model.best_iteration}")
                logger.info(f"  Val NDCG@10:    {val_metrics['val_ndcg_10']:.4f}")
                logger.info(f"  Val RMSE:       {val_metrics['val_rmse']:.4f}")

        # Select best by NDCG@10
        best_result = max(results, key=lambda x: x.val_ndcg_10)

        logger.info("=" * 60)
        logger.info("Stage 1 Complete - Best Configuration:")
        logger.info(f"  Learning Rate: {best_result.learning_rate}")
        logger.info(f"  Best Iteration: {best_result.best_iteration}")
        logger.info(f"  Val NDCG@10: {best_result.val_ndcg_10:.4f}")
        logger.info("=" * 60)

        return best_result

    def _run_stage2_diagnostics(self, best_stage1: TuningResult) -> DiagnosticResult:
        """
        Stage 2: Analyze train/val gap for overfitting/underfitting.

        Args:
            best_stage1: Best result from Stage 1

        Returns:
            DiagnosticResult with recommendations
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: DIAGNOSTICS (Train vs Val Gap)")
        logger.info("=" * 60)

        return self.diagnostics_analyzer.analyze(
            model=best_stage1.model,
            X_train=self.X_train,
            y_train=self.y_train,
            train_df=self.train_df,
            X_val=self.X_val,
            y_val=self.y_val,
            val_df=self.val_df,
        )

    def _run_stage3_secondary_tuning(
        self, best_stage1: TuningResult
    ) -> tuple[xgb.XGBRegressor, dict]:
        """
        Stage 3: Adjust complexity based on diagnostic results.

        Args:
            best_stage1: Best result from Stage 1

        Returns:
            Tuple of (best model, best params)
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: SECONDARY TUNING")
        logger.info("=" * 60)

        # Determine depths to try based on diagnostic action
        if self.diagnostic_result.stage3_action == "REDUCE_COMPLEXITY":
            depths_to_try = [3, 4, 5]
            logger.info("Trying reduced max_depth values: %s", depths_to_try)
        else:  # INCREASE_COMPLEXITY
            depths_to_try = [8, 9, 12]
            logger.info("Trying increased max_depth values: %s", depths_to_try)

        results: List[TuningResult] = []

        for depth in depths_to_try:
            logger.info(f"Training with max_depth={depth}...")

            with mlflow.start_run(run_name=f"stage3_depth_{depth}"):
                params = {
                    "max_depth": depth,
                    "learning_rate": best_stage1.learning_rate,
                    "n_estimators": self.config.n_estimators_max,
                    "colsample_bytree": self.config.colsample_bytree,
                    "early_stopping_rounds": self.config.early_stopping_rounds,
                    "objective": "reg:squarederror",
                    "random_state": self.config.random_seed,
                    "n_jobs": -1,
                }

                mlflow.log_params(params)
                mlflow.log_param("stage", "secondary_tuning")
                mlflow.log_param("action", self.diagnostic_result.stage3_action)

                model = xgb.XGBRegressor(**params)
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False,
                )

                # Evaluate
                val_metrics, _ = evaluate_model(model, self.X_val, self.y_val, self.val_df, prefix="val_")
                mlflow.log_metrics(val_metrics)
                mlflow.log_param("best_iteration", model.best_iteration)

                result = TuningResult(
                    learning_rate=best_stage1.learning_rate,
                    max_depth=depth,
                    best_iteration=model.best_iteration,
                    val_ndcg_10=val_metrics["val_ndcg_10"],
                    val_ndcg_20=val_metrics["val_ndcg_20"],
                    val_rmse=val_metrics["val_rmse"],
                    val_mae=val_metrics["val_mae"],
                    model=model,
                )
                results.append(result)

                logger.info(f"  Best iteration: {model.best_iteration}")
                logger.info(f"  Val NDCG@10:    {val_metrics['val_ndcg_10']:.4f}")

        self.stage3_results = results

        # Select best from Stage 3
        best_stage3 = max(results, key=lambda x: x.val_ndcg_10)

        logger.info("=" * 60)
        logger.info("Stage 3 Complete - Best Configuration:")
        logger.info(f"  Max Depth: {best_stage3.max_depth}")
        logger.info(f"  Val NDCG@10: {best_stage3.val_ndcg_10:.4f}")

        # Compare to Stage 1
        if best_stage3.val_ndcg_10 > best_stage1.val_ndcg_10:
            logger.info("Stage 3 improved NDCG@10!")
            best_model = best_stage3.model
            best_params = {
                "learning_rate": best_stage3.learning_rate,
                "max_depth": best_stage3.max_depth,
                "n_estimators": best_stage3.best_iteration,
                "colsample_bytree": self.config.colsample_bytree,
            }
        else:
            logger.info("Stage 1 model remains best")
            best_model = best_stage1.model
            best_params = {
                "learning_rate": best_stage1.learning_rate,
                "max_depth": best_stage1.max_depth,
                "n_estimators": best_stage1.best_iteration,
                "colsample_bytree": self.config.colsample_bytree,
            }

        logger.info("=" * 60)

        return best_model, best_params

    def _save_model(self) -> None:
        """Save the best model and its configuration."""
        models_dir = self.config.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / "xgboost_tuned.json"
        self.best_model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save feature columns
        feature_cols_path = models_dir / "feature_columns.json"
        with open(feature_cols_path, "w") as f:
            json.dump(self.feature_cols, f, indent=2)
        logger.info(f"Feature columns saved to {feature_cols_path}")

        # Evaluate for model info
        val_metrics, _ = evaluate_model(
            self.best_model, self.X_val, self.y_val, self.val_df, prefix="val_"
        )

        # Save model info
        model_info = {
            "model_name": "xgboost_tuned",
            "created_at": pd.Timestamp.now().isoformat(),
            "params": self.best_params,
            "metrics": {
                "val_ndcg_10": float(val_metrics["val_ndcg_10"]),
                "val_ndcg_20": float(val_metrics["val_ndcg_20"]),
                "val_rmse": float(val_metrics["val_rmse"]),
                "val_mae": float(val_metrics["val_mae"]),
            },
            "num_features": len(self.feature_cols),
            "tuning_summary": {
                "stage1_runs": len(self.config.learning_rates),
                "stage3_runs": len(self.stage3_results),
                "total_runs": len(self.config.learning_rates) + len(self.stage3_results),
            },
        }

        model_info_path = models_dir / "model_info.json"
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Model info saved to {model_info_path}")

    def _final_evaluation(self) -> None:
        """Perform final evaluation on test set."""
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)

        test_metrics, _ = evaluate_model(
            self.best_model, self.X_test, self.y_test, self.test_df, prefix="test_"
        )

        logger.info("Test Set Performance:")
        logger.info(f"  NDCG@10: {test_metrics['test_ndcg_10']:.4f}")
        logger.info(f"  NDCG@20: {test_metrics['test_ndcg_20']:.4f}")
        logger.info(f"  RMSE:    {test_metrics['test_rmse']:.4f}")
        logger.info(f"  MAE:     {test_metrics['test_mae']:.4f}")

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
