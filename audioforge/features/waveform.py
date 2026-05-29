from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
import torchaudio


AudioBackend = Literal["soundfile"]


@dataclass(frozen=True)
class WaveformConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    mono: bool = True
    crop_mode: Literal["center", "random", "start"] = "center"
    pad_mode: Literal["constant"] = "constant"


def load_audio(path: str | Path, backend: AudioBackend = "soundfile") -> tuple[torch.Tensor, int]:
    """Load audio as tensor shaped [channels, time].

    FSD50K is WAV, so soundfile is stable and avoids torchaudio/torchcodec loader issues.
    """

    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if backend != "soundfile":
        raise ValueError(f"Unsupported audio backend: {backend}")

    data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")

    if data.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    # soundfile returns [time, channels], PyTorch audio convention is [channels, time]
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    return waveform.float(), int(sample_rate)


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert [channels, time] waveform to mono [1, time]."""

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    if waveform.shape[0] == 1:
        return waveform

    return waveform.mean(dim=0, keepdim=True)


def resample_waveform(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Resample waveform if needed."""

    if original_sample_rate <= 0:
        raise ValueError("original_sample_rate must be positive")

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    if original_sample_rate == target_sample_rate:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )

    return resampler(waveform)


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    target_num_samples: int,
    *,
    crop_mode: Literal["center", "random", "start"] = "center",
    pad_mode: Literal["constant"] = "constant",
) -> torch.Tensor:
    """Crop or pad waveform to fixed number of samples."""

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    if target_num_samples <= 0:
        raise ValueError("target_num_samples must be positive")

    current_num_samples = waveform.shape[-1]

    if current_num_samples == target_num_samples:
        return waveform

    if current_num_samples > target_num_samples:
        extra = current_num_samples - target_num_samples

        if crop_mode == "center":
            start = extra // 2
        elif crop_mode == "start":
            start = 0
        elif crop_mode == "random":
            start = random.randint(0, extra)
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")

        end = start + target_num_samples
        return waveform[:, start:end]

    missing = target_num_samples - current_num_samples

    if pad_mode != "constant":
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    left = missing // 2
    right = missing - left

    return torch.nn.functional.pad(waveform, (left, right), mode="constant", value=0.0)


def peak_normalize_waveform(
    waveform: torch.Tensor,
    *,
    eps: float = 1e-8,
    target_peak: float = 0.95,
) -> torch.Tensor:
    """Peak normalize waveform to stable amplitude range."""

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    peak = waveform.abs().max()

    if peak < eps:
        return waveform

    return waveform / peak * target_peak


def prepare_waveform(
    path: str | Path,
    config: WaveformConfig | None = None,
    *,
    normalize_peak: bool = True,
) -> torch.Tensor:
    """Full waveform preparation.

    Pipeline:
    load audio -> mono -> resample -> fixed crop/pad -> optional peak normalize

    Returns:
        Tensor [1, target_samples] if mono=True
    """

    cfg = config or WaveformConfig()

    waveform, original_sample_rate = load_audio(path)

    if cfg.mono:
        waveform = convert_to_mono(waveform)

    waveform = resample_waveform(
        waveform,
        original_sample_rate=original_sample_rate,
        target_sample_rate=cfg.sample_rate,
    )

    target_num_samples = int(round(cfg.sample_rate * cfg.clip_seconds))

    waveform = crop_or_pad_waveform(
        waveform,
        target_num_samples=target_num_samples,
        crop_mode=cfg.crop_mode,
        pad_mode=cfg.pad_mode,
    )

    if normalize_peak:
        waveform = peak_normalize_waveform(waveform)

    return waveform.contiguous()


def get_audio_duration_seconds(path: str | Path) -> float:
    """Return audio duration in seconds without loading full waveform."""

    info = sf.info(str(path))

    if info.samplerate <= 0:
        return 0.0

    return float(info.frames) / float(info.samplerate)
