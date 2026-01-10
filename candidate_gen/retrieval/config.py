"""
Configuration for FAISS index building and retrieval.
"""

from dataclasses import dataclass


@dataclass
class FAISSConfig:
    """
    Configuration for FAISS index.

    For MovieLens (~4K items), we use a Flat index (exact search)
    which is fast enough and provides exact results.

    For larger catalogs, consider:
    - IVF: Inverted file index (approximate, faster)
    - HNSW: Hierarchical navigable small world (approximate, very fast)

    Attributes:
        index_type: Type of FAISS index
            - "flat": Exact search (default, recommended for <100K items)
            - "ivf": Inverted file (approximate, for 100K-10M items)
            - "hnsw": HNSW graph (approximate, for very fast search)

        metric: Distance metric
            - "ip": Inner product (default, for L2-normalized embeddings)
            - "l2": Euclidean distance

        nlist: Number of clusters for IVF index (default: 100)
            Higher = more accurate but slower index building

        nprobe: Number of clusters to search for IVF (default: 10)
            Higher = more accurate but slower search

        ef_construction: HNSW construction parameter (default: 200)
        ef_search: HNSW search parameter (default: 128)
    """

    index_type: str = "flat"
    metric: str = "ip"  # Inner product (cosine for L2-normalized)

    # IVF parameters
    nlist: int = 100
    nprobe: int = 10

    # HNSW parameters
    ef_construction: int = 200
    ef_search: int = 128
