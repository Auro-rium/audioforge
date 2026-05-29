from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torchaudio


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int | None = None
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    power: float = 2.0
    normalized: bool = False
    center: bool = True
    pad_mode: str = "reflect"
    log_offset: float = 1e-6
    output_shape: Literal["channel_first", "no_channel"] = "channel_first"


@dataclass(frozen=True)
class NormalizeConfig:
    enabled: bool = True
    mode: Literal["per_sample", "ast"] = "per_sample"
    eps: float = 1e-6

    # AST-style defaults commonly used by HF ASTFeatureExtractor.
    # We keep this configurable because blind normalization is how models become expensive nonsense.
    ast_mean: float = -4.2677393
    ast_std: float = 4.5689974


class LogMelExtractor:
    """Waveform → log-mel spectrogram.

    Input:
        waveform: Tensor [channels, time]

    Output:
        if output_shape == "channel_first":
            Tensor [channels, n_mels, frames]
        else:
            Tensor [n_mels, frames] for mono input
    """

    def __init__(
        self,
        config: LogMelConfig | None = None,
        normalize_config: NormalizeConfig | None = None,
    ) -> None:
        self.config = config or LogMelConfig()
        self.normalize_config = normalize_config or NormalizeConfig()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels,
            power=self.config.power,
            normalized=self.config.normalized,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform_to_logmel(
            waveform,
            extractor=self.mel,
            log_offset=self.config.log_offset,
            output_shape=self.config.output_shape,
            normalize_config=self.normalize_config,
        )


def waveform_to_logmel(
    waveform: torch.Tensor,
    *,
    extractor: torchaudio.transforms.MelSpectrogram,
    log_offset: float = 1e-6,
    output_shape: Literal["channel_first", "no_channel"] = "channel_first",
    normalize_config: NormalizeConfig | None = None,
) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    if log_offset <= 0:
        raise ValueError("log_offset must be positive")

    mel = extractor(waveform)
    logmel = torch.log(mel + log_offset)

    normalize_cfg = normalize_config or NormalizeConfig()

    if normalize_cfg.enabled:
        logmel = normalize_logmel(logmel, normalize_cfg)

    if output_shape == "channel_first":
        return logmel.contiguous()

    if output_shape == "no_channel":
        if logmel.shape[0] != 1:
            raise ValueError(
                "output_shape='no_channel' requires mono input with one channel, "
                f"got shape {tuple(logmel.shape)}"
            )
        return logmel.squeeze(0).contiguous()

    raise ValueError(f"Unsupported output_shape: {output_shape}")


def normalize_logmel(logmel: torch.Tensor, config: NormalizeConfig | None = None) -> torch.Tensor:
    """Normalize log-mel spectrogram.

    Modes:
        per_sample:
            Normalize each sample independently.
            Good for scratch CNN baseline.

        ast:
            Use global mean/std-style normalization.
            Better aligned with transformer feature preprocessing.
    """

    cfg = config or NormalizeConfig()

    if logmel.ndim not in {2, 3}:
        raise ValueError(f"Expected logmel shape [mel, frames] or [channels, mel, frames], got {tuple(logmel.shape)}")

    if cfg.mode == "per_sample":
        mean = logmel.mean()
        std = logmel.std(unbiased=False)
        return (logmel - mean) / (std + cfg.eps)

    if cfg.mode == "ast":
        return (logmel - cfg.ast_mean) / (cfg.ast_std + cfg.eps)

    raise ValueError(f"Unsupported normalization mode: {cfg.mode}")


def pad_or_crop_frames(
    spec: torch.Tensor,
    target_frames: int,
    *,
    crop_mode: Literal["center", "start"] = "center",
) -> torch.Tensor:
    """Pad or crop spectrogram along time/frame dimension.

    Supports:
        [mel, frames]
        [channels, mel, frames]
    """

    if spec.ndim not in {2, 3}:
        raise ValueError(f"Expected spec shape [mel, frames] or [channels, mel, frames], got {tuple(spec.shape)}")

    if target_frames <= 0:
        raise ValueError("target_frames must be positive")

    current_frames = spec.shape[-1]

    if current_frames == target_frames:
        return spec

    if current_frames > target_frames:
        extra = current_frames - target_frames

        if crop_mode == "center":
            start = extra // 2
        elif crop_mode == "start":
            start = 0
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")

        end = start + target_frames
        return spec[..., start:end].contiguous()

    missing = target_frames - current_frames
    left = missing // 2
    right = missing - left

    return torch.nn.functional.pad(spec, (left, right), mode="constant", value=0.0).contiguous()


def make_logmel_extractor(
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    normalize_mode: Literal["per_sample", "ast"] = "per_sample",
) -> LogMelExtractor:
    """Convenience factory used later by datasets/training scripts."""

    return LogMelExtractor(
        config=LogMelConfig(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ),
        normalize_config=NormalizeConfig(
            enabled=True,
            mode=normalize_mode,
        ),
    )