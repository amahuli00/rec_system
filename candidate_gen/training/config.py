"""
Configuration for Two-Tower model training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for Two-Tower model training.

    This config covers:
    - Model architecture (embedding_dim)
    - Training hyperparameters (batch_size, lr, epochs)
    - Loss function settings (temperature, loss type)
    - Early stopping
    - Evaluation settings
    - Paths and MLflow tracking

    Attributes:
        # Model
        embedding_dim: Dimension of user/item embeddings
        use_mlp: Whether to use MLP layers in towers

        # Training
        batch_size: Training batch size (larger = more in-batch negatives)
        learning_rate: Adam learning rate
        weight_decay: L2 regularization on embeddings
        num_epochs: Maximum training epochs
        temperature: Softmax temperature (lower = sharper distribution)
        loss_fn: Loss function ("in_batch_softmax" or "bpr")

        # Early stopping
        patience: Epochs without improvement before stopping
        min_delta: Minimum improvement to count as progress

        # Evaluation
        eval_k_values: K values for Recall@K computation
        eval_every_n_epochs: How often to evaluate on validation set

        # Device and reproducibility
        device: Training device ("cpu" or "cuda")
        seed: Random seed for reproducibility

        # Paths
        data_dir: Directory containing processed data
        output_dir: Directory to save model and artifacts

        # MLflow
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment: MLflow experiment name
        run_name: Optional run name (auto-generated if None)
    """

    # Model
    embedding_dim: int = 128
    use_mlp: bool = False

    # Training
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    num_epochs: int = 30
    temperature: float = 0.1
    loss_fn: str = "in_batch_softmax"  # or "bpr"

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

    # Evaluation
    eval_k_values: List[int] = field(default_factory=lambda: [10, 50, 100])
    eval_every_n_epochs: int = 5

    # Device and reproducibility
    device: str = "cpu"
    seed: int = 42

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("candidate_gen/artifacts/data"))
    output_dir: Path = field(default_factory=lambda: Path("candidate_gen/artifacts/models"))

    # MLflow
    mlflow_tracking_uri: str = "file:///Users/ashishmahuli/Desktop/rec_system/mlruns"
    mlflow_experiment: str = "candidate_generation"
    run_name: Optional[str] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "embedding_dim": self.embedding_dim,
            "use_mlp": self.use_mlp,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "temperature": self.temperature,
            "loss_fn": self.loss_fn,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "eval_k_values": self.eval_k_values,
            "eval_every_n_epochs": self.eval_every_n_epochs,
            "device": self.device,
            "seed": self.seed,
        }
