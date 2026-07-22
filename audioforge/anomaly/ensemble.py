from __future__ import annotations

import numpy as np


def minmax_normalize(scores: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low + eps)


def ensemble_scores(*scores: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    if not scores:
        raise ValueError("at least one score array is required")
    arrays = [minmax_normalize(item) for item in scores]
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("all score arrays must have equal length")
    if weights is None:
        weights = [1.0] * len(arrays)
    if len(weights) != len(arrays) or sum(weights) <= 0:
        raise ValueError("weights must match scores and have a positive sum")
    return np.average(np.stack(arrays), axis=0, weights=np.asarray(weights)).astype(np.float32)
