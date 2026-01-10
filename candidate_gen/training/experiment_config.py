"""
Experiment configuration for hyperparameter sweeps.

Provides structured configuration for running grid search experiments
with MLflow tracking.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import TrainingConfig


@dataclass
class ExperimentConfig:
    """
    Configuration for hyperparameter experiments.

    Supports grid search over specified parameters with multiple seeds
    for variance estimation.

    Example YAML:
        name: sweep_embedding_dim
        description: Find optimal embedding dimension

        base_config:
          batch_size: 1024
          learning_rate: 0.001
          num_epochs: 20

        sweep_params:
          embedding_dim: [32, 64, 128, 256]

        num_seeds: 3

    Attributes:
        name: Experiment name (used for MLflow experiment)
        description: Human-readable description
        base_config: Default training configuration
        sweep_params: Parameters to sweep (grid search)
        num_seeds: Number of random seeds per config (for variance)
        output_dir: Base output directory for all runs
    """

    name: str
    description: str = ""
    base_config: TrainingConfig = field(default_factory=TrainingConfig)
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)
    num_seeds: int = 1
    output_dir: Path = field(default_factory=lambda: Path("candidate_gen/artifacts/experiments"))

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate sweep params exist in TrainingConfig
        valid_params = set(self.base_config.to_dict().keys())
        for param in self.sweep_params:
            if param not in valid_params and param != "seed":
                raise ValueError(
                    f"Invalid sweep parameter: {param}. "
                    f"Valid parameters: {valid_params}"
                )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentConfig":
        """
        Load experiment configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            ExperimentConfig instance
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Parse base_config
        base_config_dict = config_dict.get("base_config", {})
        if "data_dir" in base_config_dict:
            base_config_dict["data_dir"] = Path(base_config_dict["data_dir"])
        if "output_dir" in base_config_dict:
            base_config_dict["output_dir"] = Path(base_config_dict["output_dir"])

        base_config = TrainingConfig(**base_config_dict)

        # Parse output_dir
        output_dir = config_dict.get("output_dir", "candidate_gen/artifacts/experiments")
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        return cls(
            name=config_dict.get("name", "experiment"),
            description=config_dict.get("description", ""),
            base_config=base_config,
            sweep_params=config_dict.get("sweep_params", {}),
            num_seeds=config_dict.get("num_seeds", 1),
            output_dir=output_dir,
        )

    def generate_configs(self) -> List[TrainingConfig]:
        """
        Generate all training configurations for the sweep.

        Returns:
            List of TrainingConfig objects for each combination
        """
        if not self.sweep_params:
            # No sweep, just return base config with different seeds
            return [self._create_config({}, seed) for seed in range(self.num_seeds)]

        # Generate all combinations
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        combinations = list(itertools.product(*param_values))

        configs = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            for seed_offset in range(self.num_seeds):
                seed = self.base_config.seed + seed_offset
                config = self._create_config(param_dict, seed)
                configs.append(config)

        return configs

    def _create_config(self, param_overrides: Dict[str, Any], seed: int) -> TrainingConfig:
        """
        Create a TrainingConfig with specified overrides.

        Args:
            param_overrides: Parameter values to override
            seed: Random seed

        Returns:
            TrainingConfig instance
        """
        # Start with base config as dict
        config_dict = self.base_config.to_dict()

        # Apply overrides
        for param, value in param_overrides.items():
            config_dict[param] = value

        config_dict["seed"] = seed

        # Create run name
        override_str = "_".join(f"{k}{v}" for k, v in param_overrides.items())
        run_name = f"{override_str}_seed{seed}" if override_str else f"seed{seed}"
        config_dict["run_name"] = run_name

        # Set output dir for this specific run
        run_output_dir = self.output_dir / self.name / run_name
        config_dict["output_dir"] = run_output_dir

        # Set experiment name
        config_dict["mlflow_experiment"] = f"candidate_gen_{self.name}"

        # Paths need special handling
        config_dict["data_dir"] = self.base_config.data_dir

        return TrainingConfig(**config_dict)

    def get_num_runs(self) -> int:
        """Get total number of runs in this experiment."""
        if not self.sweep_params:
            return self.num_seeds

        num_combinations = 1
        for values in self.sweep_params.values():
            num_combinations *= len(values)

        return num_combinations * self.num_seeds
