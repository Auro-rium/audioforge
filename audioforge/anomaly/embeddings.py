from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audioforge.features.logmel import LogMelExtractor, make_logmel_extractor
from audioforge.features.waveform import WaveformConfig, prepare_waveform


@dataclass(frozen=True)
class EmbeddingConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64


class LogMelEmbeddingExtractor:
    """Deterministic, model-free baseline embedding for machine audio.

    Each clip becomes the concatenation of per-mel mean and standard deviation
    over time. This is intentionally transparent and provides a stable baseline
    for comparing anomaly scoring methods before adding pretrained encoders.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.extractor: LogMelExtractor = make_logmel_extractor(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            normalize_mode="per_sample",
        )

    def __call__(self, audio_path: str | Path) -> np.ndarray:
        waveform = prepare_waveform(
            audio_path,
            WaveformConfig(
                sample_rate=self.config.sample_rate,
                clip_seconds=self.config.clip_seconds,
                crop_mode="center",
            ),
        )
        with torch.no_grad():
            logmel = self.extractor(waveform).squeeze(0)
            vector = torch.cat((logmel.mean(dim=-1), logmel.std(dim=-1, unbiased=False)))
        return vector.cpu().numpy().astype(np.float32)
