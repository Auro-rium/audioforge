from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from audioforge.anomaly.embeddings import EmbeddingConfig, LogMelEmbeddingExtractor
from audioforge.anomaly.ensemble import ensemble_scores
from audioforge.anomaly.knn import fit_knn
from audioforge.anomaly.mahalanobis import MahalanobisScorer
from audioforge.data.dcase import read_dcase_manifest
from audioforge.evaluation.dcase_metrics import DCASEPrediction, write_dcase_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit DCASE classical anomaly baselines.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    args = parser.parse_args()

    config = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        if not isinstance(config, dict):
            raise SystemExit("DCASE config must be a YAML mapping")

    manifest_value = args.manifest or config.get("manifest")
    output_value = args.output or config.get("output")
    if not manifest_value or not output_value:
        raise SystemExit("Provide --manifest and --output, directly or through --config")
    manifest = Path(manifest_value)
    output = Path(output_value)
    n_neighbors = int(args.n_neighbors if args.n_neighbors is not None else config.get("n_neighbors", 5))
    sample_rate = int(args.sample_rate if args.sample_rate is not None else config.get("sample_rate", 16_000))
    clip_seconds = float(args.clip_seconds if args.clip_seconds is not None else config.get("clip_seconds", 10.0))
    rows = read_dcase_manifest(manifest)
    normal_rows = [row for row in rows if row.label == "normal" and row.split == "train"]
    if len(normal_rows) < 2:
        raise SystemExit("DCASE manifest needs at least two normal training rows")
    extractor = LogMelEmbeddingExtractor(
        EmbeddingConfig(sample_rate=sample_rate, clip_seconds=clip_seconds)
    )
    embeddings = np.stack([extractor(row.path) for row in normal_rows])
    knn = fit_knn(embeddings, n_neighbors=n_neighbors)
    mahalanobis = MahalanobisScorer().fit(embeddings)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"),
        embeddings=embeddings,
        mean=mahalanobis.mean,
        precision=mahalanobis.precision,
    )
    metadata = {
        "manifest": str(manifest),
        "normal_training_rows": len(normal_rows),
        "embedding_dim": int(embeddings.shape[1]),
        "n_neighbors": n_neighbors,
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "methods": ["knn", "mahalanobis", "ensemble"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_rows = [row for row in rows if row.split in {"test", "eval"}]
    if all_rows:
        test_embeddings = np.stack([extractor(row.path) for row in all_rows])
        knn_scores = knn.score(test_embeddings)
        mahalanobis_scores = mahalanobis.score(test_embeddings)
        scores = ensemble_scores(knn_scores, mahalanobis_scores)
        prediction_path = output.with_name(f"{output.stem}_predictions.csv")
        write_dcase_predictions(
            prediction_path,
            (
                DCASEPrediction(
                    path=row.path,
                    machine_type=row.machine_type,
                    section=row.section,
                    domain=row.domain,
                    y_true=1 if row.label == "anomaly" else 0,
                    anomaly_score=float(score),
                )
                for row, score in zip(all_rows, scores)
            ),
        )


if __name__ == "__main__":
    main()
