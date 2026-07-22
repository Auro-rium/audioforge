from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoFeatureExtractor

from audioforge.features.logmel import make_logmel_extractor
from audioforge.features.waveform import WaveformConfig, prepare_waveform
from audioforge.models.ast import create_ast_classifier
from audioforge.models.scratch_cnn import create_scratch_cnn


@dataclass(frozen=True)
class EventPrediction:
    label: str
    score: float


class EventPredictor:
    """Load a trained FSD50K checkpoint and predict sound events.

    The checkpoint format is the one emitted by ``FSD50KTrainer``.  Model and
    preprocessing settings are read from its embedded ``extra.config`` payload;
    only the label map remains an explicit deployment artifact.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        label_map_path: str | Path,
        *,
        device: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.label_map_path = Path(label_map_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {self.label_map_path}")

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("Checkpoint must contain a model_state_dict")

        extra = payload.get("extra", {})
        config = extra.get("config", {}) if isinstance(extra, dict) else {}
        if not isinstance(config, dict):
            config = {}

        with self.label_map_path.open("r", encoding="utf-8") as file:
            label_payload = json.load(file)
        raw_map = label_payload.get("id_to_label")
        if not isinstance(raw_map, dict):
            raise ValueError("Label map must contain an id_to_label mapping")
        self.id_to_label = {int(index): str(label) for index, label in raw_map.items()}

        self.model_name = str(config.get("model_name", "scratch_cnn"))
        self.sample_rate = int(config.get("sample_rate", 16_000))
        self.clip_seconds = float(config.get("clip_seconds", 10.0))
        self.threshold = float(
            threshold if threshold is not None else config.get("threshold", 0.5)
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self._build_model(config)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()

        self.logmel = None
        self.ast_feature_extractor = None
        if self.model_name == "scratch_cnn":
            self.logmel = make_logmel_extractor(
                sample_rate=self.sample_rate,
                n_fft=int(config.get("n_fft", 1024)),
                hop_length=int(config.get("hop_length", 512)),
                n_mels=int(config.get("n_mels", 128)),
                normalize_mode=str(config.get("normalize_mode", "per_sample")),  # type: ignore[arg-type]
            )
        elif self.model_name == "ast":
            model_name = config.get(
                "pretrained_name_or_path", "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            self.ast_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def _build_model(self, config: dict[str, Any]) -> torch.nn.Module:
        num_labels = int(config.get("num_labels", len(self.id_to_label)))
        if num_labels != len(self.id_to_label):
            raise ValueError("Checkpoint num_labels does not match the label map")
        if self.model_name == "scratch_cnn":
            return create_scratch_cnn(
                num_labels=num_labels,
                base_channels=int(config.get("base_channels", 32)),
                dropout=float(config.get("dropout", 0.2)),
            )
        if self.model_name == "ast":
            return create_ast_classifier(
                pretrained_name_or_path=config.get(
                    "pretrained_name_or_path", "MIT/ast-finetuned-audioset-10-10-0.4593"
                ),
                num_labels=num_labels,
                dropout=float(config.get("dropout", 0.1)),
            )
        raise ValueError(f"Unsupported checkpoint model_name: {self.model_name}")

    @torch.inference_mode()
    def predict(self, audio_path: str | Path, *, top_k: int = 5) -> list[EventPrediction]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        waveform = prepare_waveform(
            audio_path,
            WaveformConfig(
                sample_rate=self.sample_rate,
                clip_seconds=self.clip_seconds,
                crop_mode="center",
            ),
        )
        if self.model_name == "scratch_cnn":
            assert self.logmel is not None
            inputs = self.logmel(waveform).unsqueeze(0)
        else:
            assert self.ast_feature_extractor is not None
            features = self.ast_feature_extractor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = features["input_values"]

        probabilities = torch.sigmoid(self.model(inputs.to(self.device)))[0].cpu()
        values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
        return [
            EventPrediction(
                label=self.id_to_label[int(index)],
                score=float(value),
            )
            for value, index in zip(values, indices)
            if float(value) >= self.threshold
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FSD50K sound events.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = EventPredictor(
        args.checkpoint,
        args.label_map,
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps([item.__dict__ for item in predictor.predict(args.audio, top_k=args.top_k)]))


if __name__ == "__main__":
    main()
