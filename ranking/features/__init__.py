"""
Feature engineering package for the ranking model.

This package contains modules for building GBDT ranking features from
MovieLens data splits. Features are computed from training data only
to prevent data leakage.

Modules:
    config: Configuration dataclasses (FeatureConfig, FeatureMetadata)
    builders: FeatureBuilder class for the complete pipeline

CLI Usage:
    python -m ranking.features.build_features

Programmatic Usage:
    from ranking.features import FeatureBuilder, FeatureConfig

    config = FeatureConfig(
        data_dir=Path("data/splits"),
        output_dir=Path("ranking/features")
    )
    builder = FeatureBuilder(config)
    train_features, val_features, test_features = builder.run()
"""

from .config import FeatureConfig, FeatureMetadata
from .builders import FeatureBuilder

__all__ = [
    "FeatureConfig",
    "FeatureMetadata",
    "FeatureBuilder",
]
