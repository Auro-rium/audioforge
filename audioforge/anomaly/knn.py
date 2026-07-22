from __future__ import annotations

import numpy as np

from audioforge.anomaly.memory_bank import MemoryBank


def fit_knn(embeddings: np.ndarray, *, n_neighbors: int = 5) -> MemoryBank:
    return MemoryBank(n_neighbors=n_neighbors).fit(embeddings)


def knn_scores(model: MemoryBank, embeddings: np.ndarray) -> np.ndarray:
    return model.score(embeddings)
