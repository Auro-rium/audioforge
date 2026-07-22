from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MahalanobisScorer:
    regularization: float = 1e-4
    mean: np.ndarray | None = None
    precision: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "MahalanobisScorer":
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("at least two 2D normal embeddings are required")
        self.mean = values.mean(axis=0)
        covariance = np.cov(values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0]) * self.regularization
        self.precision = np.linalg.pinv(covariance)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise RuntimeError("MahalanobisScorer must be fitted before scoring")
        delta = np.asarray(embeddings, dtype=np.float64) - self.mean
        distances = np.einsum("ni,ij,nj->n", delta, self.precision, delta)
        return np.sqrt(np.maximum(distances, 0.0)).astype(np.float32)
