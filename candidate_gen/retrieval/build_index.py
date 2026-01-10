"""
CLI entry point for building FAISS index.

Usage:
    python -m candidate_gen.retrieval.build_index --model-dir candidate_gen/artifacts/models
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from ..data import IDMapper
from ..model import ModelConfig, TwoTowerModel
from .config import FAISSConfig
from .index_builder import EmbeddingMaterializer, FAISSIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    faiss_config: FAISSConfig,
    device: str = "cpu",
) -> None:
    """
    Build FAISS index from trained model.

    Steps:
    1. Load trained model
    2. Materialize user and item embeddings
    3. Build FAISS index from item embeddings
    4. Save all artifacts

    Args:
        model_dir: Directory containing trained model
        data_dir: Directory containing ID mappings
        output_dir: Base output directory
        faiss_config: FAISS configuration
        device: Computation device
    """
    logger.info("=" * 60)
    logger.info("BUILDING FAISS INDEX")
    logger.info("=" * 60)

    # =========================================================================
    # Stage 1: Load model and ID mapper
    # =========================================================================
    logger.info("Loading model and ID mappings...")

    # Load model info
    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)

    num_users = model_info["num_users"]
    num_items = model_info["num_items"]
    embedding_dim = model_info["embedding_dim"]
    use_mlp = model_info.get("use_mlp", False)

    logger.info(f"Model: {num_users:,} users, {num_items:,} items, dim={embedding_dim}")

    # Create model
    model_config = ModelConfig(embedding_dim=embedding_dim, use_mlp=use_mlp)
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        config=model_config,
    ).to(device)

    # Load weights
    model.load_state_dict(
        torch.load(model_dir / "two_tower_model.pt", map_location=device)
    )
    model.eval()
    logger.info("Model loaded successfully")

    # Load ID mapper
    id_mapper = IDMapper.load(data_dir)

    # =========================================================================
    # Stage 2: Materialize embeddings
    # =========================================================================
    logger.info("Materializing embeddings...")

    embeddings_dir = output_dir / "embeddings"
    materializer = EmbeddingMaterializer(model, num_users, num_items, device)
    materializer.materialize(embeddings_dir)

    # =========================================================================
    # Stage 3: Build FAISS index
    # =========================================================================
    logger.info("Building FAISS index...")

    # Load item embeddings
    item_embeddings = np.load(embeddings_dir / "item_embeddings.npy")

    # Build index
    builder = FAISSIndexBuilder(faiss_config)
    index = builder.build(item_embeddings)

    # Save index
    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    builder.save(index, index_dir / "item_index.faiss")

    # Save index metadata
    index_metadata = {
        "index_type": faiss_config.index_type,
        "metric": faiss_config.metric,
        "num_vectors": index.ntotal,
        "embedding_dim": embedding_dim,
    }
    with open(index_dir / "index_metadata.json", "w") as f:
        json.dump(index_metadata, f, indent=2)

    # =========================================================================
    # Stage 4: Verification
    # =========================================================================
    logger.info("Verifying index...")

    # Test retrieval
    test_user_idx = 0
    user_emb = materializer._materialize_embeddings(
        model.get_user_embedding, 1, 1
    )
    distances, indices = index.search(user_emb, 10)

    logger.info(f"Test retrieval for user 0:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        movie_id = id_mapper.idx_to_item.get(idx, "unknown")
        logger.info(f"  {rank}. movie_id={movie_id}, score={dist:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Index building complete!")
    logger.info("=" * 60)
    logger.info(f"Embeddings: {embeddings_dir}")
    logger.info(f"Index: {index_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index from trained Two-Tower model"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("candidate_gen/artifacts/models"),
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("candidate_gen/artifacts/data"),
        help="Directory containing ID mappings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("candidate_gen/artifacts"),
        help="Base output directory",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type (default: flat)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (default: cpu)",
    )

    args = parser.parse_args()

    faiss_config = FAISSConfig(index_type=args.index_type)

    build_index(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        faiss_config=faiss_config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
