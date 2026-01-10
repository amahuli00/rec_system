"""
Experiment runner for hyperparameter sweeps.

Runs grid search experiments with MLflow tracking and results aggregation.

Usage:
    python -m candidate_gen.training.experiment_runner --config experiments/sweep_embedding_dim.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import TrainingConfig
from .experiment_config import ExperimentConfig
from .trainer import TwoTowerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Run hyperparameter experiments with grid search.

    Features:
    - Grid search over specified parameters
    - Multiple seeds for variance estimation
    - MLflow tracking for all runs
    - Results aggregation and comparison
    - Best config selection

    Example:
        config = ExperimentConfig.from_yaml("experiments/sweep_embedding_dim.yaml")
        runner = ExperimentRunner(config)
        results_df = runner.run()
        best_config = runner.get_best_config(metric="recall@50")
    """

    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize the experiment runner.

        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config
        self.results: List[Dict] = []

    def run(self) -> pd.DataFrame:
        """
        Run all experiment configurations.

        Returns:
            DataFrame with results for all runs
        """
        configs = self.config.generate_configs()
        num_runs = len(configs)

        logger.info("=" * 60)
        logger.info(f"EXPERIMENT: {self.config.name}")
        logger.info("=" * 60)
        logger.info(f"Description: {self.config.description}")
        logger.info(f"Sweep parameters: {list(self.config.sweep_params.keys())}")
        logger.info(f"Total runs: {num_runs}")
        logger.info("=" * 60)

        for i, config in enumerate(configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"RUN {i}/{num_runs}: {config.run_name}")
            logger.info(f"{'='*60}")

            try:
                # Train model
                trainer = TwoTowerTrainer(config)
                model, final_metrics = trainer.run()

                # Record results
                result = {
                    "run_name": config.run_name,
                    "seed": config.seed,
                    **{k: v for k, v in config.to_dict().items()
                       if k in self.config.sweep_params or k in ["embedding_dim", "learning_rate", "temperature", "batch_size"]},
                    **{f"val_{k}": v for k, v in final_metrics.items()},
                    "best_epoch": trainer.best_epoch,
                    "status": "success",
                }
                self.results.append(result)

            except Exception as e:
                logger.error(f"Run failed: {e}")
                result = {
                    "run_name": config.run_name,
                    "seed": config.seed,
                    "status": "failed",
                    "error": str(e),
                }
                self.results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(self.results)

        # Save results
        self._save_results(results_df)

        # Print summary
        self._print_summary(results_df)

        return results_df

    def _save_results(self, results_df: pd.DataFrame) -> None:
        """Save results to CSV."""
        output_dir = self.config.output_dir / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to: {results_path}")

    def _print_summary(self, results_df: pd.DataFrame) -> None:
        """Print experiment summary."""
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)

        # Filter successful runs
        successful = results_df[results_df["status"] == "success"]
        if len(successful) == 0:
            logger.warning("No successful runs!")
            return

        # Group by sweep params (excluding seed)
        sweep_params = list(self.config.sweep_params.keys())
        if sweep_params:
            # Aggregate by sweep params
            agg_cols = ["val_recall@10", "val_recall@50", "val_recall@100"]
            agg_cols = [c for c in agg_cols if c in successful.columns]

            if agg_cols:
                summary = successful.groupby(sweep_params)[agg_cols].agg(["mean", "std"])
                logger.info("\nResults by configuration:")
                logger.info(summary.to_string())

        # Best configuration
        best_metric = "val_recall@50"
        if best_metric in successful.columns:
            best_idx = successful[best_metric].idxmax()
            best_run = successful.loc[best_idx]

            logger.info(f"\nBest configuration ({best_metric}):")
            for param in sweep_params + ["seed"]:
                if param in best_run:
                    logger.info(f"  {param}: {best_run[param]}")
            logger.info(f"  {best_metric}: {best_run[best_metric]:.4f}")

    def get_best_config(
        self, metric: str = "recall@50"
    ) -> Optional[TrainingConfig]:
        """
        Get the best training configuration based on metric.

        Args:
            metric: Metric to optimize (without 'val_' prefix)

        Returns:
            TrainingConfig for the best run, or None if no successful runs
        """
        if not self.results:
            return None

        results_df = pd.DataFrame(self.results)
        successful = results_df[results_df["status"] == "success"]

        if len(successful) == 0:
            return None

        metric_col = f"val_{metric}"
        if metric_col not in successful.columns:
            logger.warning(f"Metric {metric_col} not found in results")
            return None

        best_idx = successful[metric_col].idxmax()
        best_run = successful.loc[best_idx]

        # Reconstruct config
        config = TrainingConfig(
            embedding_dim=int(best_run.get("embedding_dim", 128)),
            learning_rate=float(best_run.get("learning_rate", 1e-3)),
            temperature=float(best_run.get("temperature", 0.1)),
            batch_size=int(best_run.get("batch_size", 1024)),
            seed=int(best_run.get("seed", 42)),
            data_dir=self.config.base_config.data_dir,
            output_dir=self.config.output_dir / self.config.name / "best",
        )

        return config

    def plot_results(
        self,
        param: str,
        metric: str = "recall@50",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot metric vs parameter.

        Args:
            param: Parameter to plot on x-axis
            metric: Metric to plot on y-axis (without 'val_' prefix)
            save_path: Path to save figure (optional)
        """
        if not self.results:
            logger.warning("No results to plot")
            return

        results_df = pd.DataFrame(self.results)
        successful = results_df[results_df["status"] == "success"]

        metric_col = f"val_{metric}"
        if metric_col not in successful.columns or param not in successful.columns:
            logger.warning(f"Cannot plot: {param} or {metric_col} not in results")
            return

        # Group by param and compute mean/std
        grouped = successful.groupby(param)[metric_col].agg(["mean", "std"])

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        x = grouped.index
        y = grouped["mean"]
        yerr = grouped["std"]

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=5, capthick=2)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {param}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter experiments for Two-Tower model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after experiment",
    )

    args = parser.parse_args()

    # Load experiment config
    experiment_config = ExperimentConfig.from_yaml(args.config)

    # Run experiments
    runner = ExperimentRunner(experiment_config)
    results_df = runner.run()

    # Generate plots if requested
    if args.plot and experiment_config.sweep_params:
        for param in experiment_config.sweep_params:
            plot_path = (
                experiment_config.output_dir
                / experiment_config.name
                / f"{param}_vs_recall50.png"
            )
            runner.plot_results(param=param, metric="recall@50", save_path=plot_path)

    # Print best config
    best_config = runner.get_best_config()
    if best_config:
        logger.info("\nTo train with best config:")
        logger.info(
            f"  python -m candidate_gen.training.train_model "
            f"--embedding-dim {best_config.embedding_dim} "
            f"--lr {best_config.learning_rate} "
            f"--temperature {best_config.temperature}"
        )


if __name__ == "__main__":
    main()
