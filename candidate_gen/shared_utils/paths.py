"""
Centralized path definitions for the candidate generation system.

All path constants are defined here to avoid duplication and ensure
consistency across modules.
"""

from pathlib import Path

# Project root (rec_system/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data" / "splits"

# Candidate generation directories
CANDIDATE_GEN_DIR = PROJECT_ROOT / "candidate_gen"
ARTIFACTS_DIR = CANDIDATE_GEN_DIR / "artifacts"

# Specific artifact directories
PREPARED_DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
INDEX_DIR = ARTIFACTS_DIR / "index"
