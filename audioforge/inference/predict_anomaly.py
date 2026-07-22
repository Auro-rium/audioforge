from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audioforge.anomaly.embeddings import EmbeddingConfig, LogMelEmbeddingExtractor
from audioforge.anomaly.ensemble import ensemble_scores
from audioforge.anomaly.memory_bank import MemoryBank
from audioforge.anomaly.mahalanobis import MahalanobisScorer


def predict_anomaly(
    audio_path: str | Path,
    model_path: str | Path,
    *,
    method: str = "ensemble",
) -> float:
    """Score one audio file using a model produced by ``train_dcase``."""
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = np.load(model_path)
    extractor = LogMelEmbeddingExtractor(
        EmbeddingConfig(
            sample_rate=int(metadata.get("sample_rate", 16_000)),
            clip_seconds=float(metadata.get("clip_seconds", 10.0)),
        )
    )
    embedding = extractor(audio_path)[None, :]

    knn = MemoryBank(n_neighbors=int(metadata.get("n_neighbors", 5))).fit(artifact["embeddings"])
    mahalanobis = MahalanobisScorer(
        mean=artifact["mean"],
        precision=artifact["precision"],
    )
    knn_score = knn.score(embedding)
    mahalanobis_score = mahalanobis.score(embedding)
    if method == "knn":
        return float(knn_score[0])
    if method == "mahalanobis":
        return float(mahalanobis_score[0])
    if method == "ensemble":
        return float(ensemble_scores(knn_score, mahalanobis_score)[0])
    raise ValueError("method must be knn, mahalanobis, or ensemble")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an audio file for DCASE anomaly.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="Path to the .npz artifact")
    parser.add_argument("--method", choices=["knn", "mahalanobis", "ensemble"], default="ensemble")
    args = parser.parse_args()
    print(json.dumps({"audio": str(args.audio), "method": args.method, "anomaly_score": predict_anomaly(args.audio, args.model, method=args.method)}))


if __name__ == "__main__":
    main()
