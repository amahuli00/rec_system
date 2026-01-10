"""
Centralized path definitions for the ranking system.

All path constants are defined here to avoid duplication and ensure
consistency across modules.
"""

from pathlib import Path

# Project root (rec_system/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data" / "splits"

# Ranking directories
RANKING_DIR = PROJECT_ROOT / "ranking"
MODELS_DIR = RANKING_DIR / "models"
FEATURES_DIR = RANKING_DIR / "features"
