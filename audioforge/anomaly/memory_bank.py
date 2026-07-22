from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class MemoryBank:
    """Nearest-neighbour anomaly scorer fitted on known-normal embeddings."""

    n_neighbors: int = 5
    embeddings: np.ndarray | None = None
    index: NearestNeighbors | None = None

    def fit(self, embeddings: np.ndarray) -> "MemoryBank":
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        self.embeddings = values
        self.index = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(values)))
        self.index.fit(values)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.index is None:
            raise RuntimeError("MemoryBank must be fitted before scoring")
        distances, _ = self.index.kneighbors(np.asarray(embeddings, dtype=np.float32))
        return distances.mean(axis=1).astype(np.float32)
