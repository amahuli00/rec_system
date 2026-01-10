"""
Dataset utilities for Two-Tower candidate generation.

This module provides:
- IDMapper: Bidirectional mappings between original IDs and contiguous indices
- TwoTowerDataset: PyTorch Dataset for training with in-batch negatives
- build_user_positive_items: Build user → positive items lookup for evaluation
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IDMapper:
    """
    Manages bidirectional ID mappings for users and items.

    PyTorch embedding layers require contiguous integer indices starting from 0.
    This class creates and manages mappings:
    - user_id → user_idx (0 to num_users-1)
    - movie_id → item_idx (0 to num_items-1)

    Example:
        mapper = IDMapper.from_dataframe(train_ratings)
        user_idx = mapper.user_to_idx[user_id]
        movie_id = mapper.idx_to_item[item_idx]
    """

    def __init__(
        self,
        user_to_idx: Dict[int, int],
        item_to_idx: Dict[int, int],
    ):
        """
        Initialize with pre-built mappings.

        Args:
            user_to_idx: Mapping from original user_id to index
            item_to_idx: Mapping from original movie_id to index
        """
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx

        # Build reverse mappings
        self.idx_to_user = {v: k for k, v in user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}

    @property
    def num_users(self) -> int:
        """Number of unique users."""
        return len(self.user_to_idx)

    @property
    def num_items(self) -> int:
        """Number of unique items."""
        return len(self.item_to_idx)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "IDMapper":
        """
        Build mappings from a DataFrame with user_id and movie_id columns.

        Uses ALL unique users and items in the DataFrame, sorted for
        deterministic ordering.

        Args:
            df: DataFrame with 'user_id' and 'movie_id' columns

        Returns:
            IDMapper instance
        """
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["movie_id"].unique())

        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

        return cls(user_to_idx, item_to_idx)

    def verify(self) -> bool:
        """
        Verify that mappings are valid (bijective and contiguous).

        Returns:
            True if valid, raises AssertionError otherwise
        """
        # Check bijective
        assert len(self.user_to_idx) == len(self.idx_to_user), "User mapping not bijective"
        assert len(self.item_to_idx) == len(self.idx_to_item), "Item mapping not bijective"

        # Check contiguous
        assert set(self.user_to_idx.values()) == set(range(self.num_users)), (
            "User indices not contiguous"
        )
        assert set(self.item_to_idx.values()) == set(range(self.num_items)), (
            "Item indices not contiguous"
        )

        return True

    def save(self, output_dir: Path) -> None:
        """
        Save mappings to JSON files.

        Args:
            output_dir: Directory to save mapping files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert int keys to strings for JSON
        user_mapping = {str(k): v for k, v in self.user_to_idx.items()}
        item_mapping = {str(k): v for k, v in self.item_to_idx.items()}

        with open(output_dir / "user_to_idx.json", "w") as f:
            json.dump(user_mapping, f)

        with open(output_dir / "item_to_idx.json", "w") as f:
            json.dump(item_mapping, f)

        logger.info(f"Saved ID mappings to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path) -> "IDMapper":
        """
        Load mappings from JSON files.

        Args:
            input_dir: Directory containing mapping files

        Returns:
            IDMapper instance
        """
        with open(input_dir / "user_to_idx.json") as f:
            user_mapping = json.load(f)

        with open(input_dir / "item_to_idx.json") as f:
            item_mapping = json.load(f)

        # Convert string keys back to int
        user_to_idx = {int(k): v for k, v in user_mapping.items()}
        item_to_idx = {int(k): v for k, v in item_mapping.items()}

        return cls(user_to_idx, item_to_idx)


class TwoTowerDataset(Dataset):
    """
    PyTorch Dataset for Two-Tower model training with in-batch negatives.

    Each sample contains only:
    - user_idx: Internal user index
    - pos_item_idx: Positive item index

    Negatives are not explicitly sampled here. Instead, all other items
    in the same training batch serve as negatives (in-batch negatives).
    This is more efficient and prevents overfitting to static negatives.

    Example:
        dataset = TwoTowerDataset(train_positive_df, id_mapper)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        for batch in loader:
            user_ids = batch['user_idx']      # [1024]
            pos_items = batch['pos_item_idx'] # [1024]
            # All 1024 items serve as negatives for each user
    """

    def __init__(
        self,
        positive_df: pd.DataFrame,
        id_mapper: IDMapper,
        verbose: bool = True,
    ):
        """
        Build dataset from positive interactions.

        Args:
            positive_df: DataFrame with positive interactions (rating >= threshold)
            id_mapper: IDMapper for converting IDs to indices
            verbose: Whether to show progress bar
        """
        self.samples: List[Dict[str, int]] = []

        iterator = positive_df.iterrows()
        if verbose:
            iterator = tqdm(iterator, total=len(positive_df), desc="Building dataset")

        for _, row in iterator:
            user_idx = id_mapper.user_to_idx.get(row["user_id"])
            item_idx = id_mapper.item_to_idx.get(row["movie_id"])

            # Skip if user or item not in mapping (shouldn't happen for train)
            if user_idx is not None and item_idx is not None:
                self.samples.append({"user_idx": user_idx, "pos_item_idx": item_idx})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "user_idx": torch.tensor(sample["user_idx"], dtype=torch.long),
            "pos_item_idx": torch.tensor(sample["pos_item_idx"], dtype=torch.long),
        }


def build_user_positive_items(
    positive_df: pd.DataFrame,
    id_mapper: IDMapper,
) -> Dict[int, Set[int]]:
    """
    Build a dictionary mapping user index to set of positive item indices.

    This is used for:
    - Evaluation: Computing Recall@K
    - (Not used here): Explicit negative sampling

    Args:
        positive_df: DataFrame with positive interactions
        id_mapper: IDMapper for converting IDs to indices

    Returns:
        Dict mapping user_idx to set of positive item_idx
    """
    user_positive_items: Dict[int, Set[int]] = defaultdict(set)

    for _, row in positive_df.iterrows():
        user_idx = id_mapper.user_to_idx.get(row["user_id"])
        item_idx = id_mapper.item_to_idx.get(row["movie_id"])

        if user_idx is not None and item_idx is not None:
            user_positive_items[user_idx].add(item_idx)

    return dict(user_positive_items)
