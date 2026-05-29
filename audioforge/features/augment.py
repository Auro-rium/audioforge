from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WaveformAugmentConfig:
    enabled: bool = True
    gain_prob: float = 0.3
    noise_prob: float = 0.3
    shift_prob: float = 0.3

    min_gain: float = 0.7
    max_gain: float = 1.3

    noise_std_min: float = 0.001
    noise_std_max: float = 0.015

    max_shift_fraction: float = 0.1


@dataclass(frozen=True)
class SpecAugmentConfig:
    enabled: bool = True
    freq_mask_prob: float = 0.3
    time_mask_prob: float = 0.3
    max_freq_mask_width: int = 16
    max_time_mask_width: int = 32
    num_freq_masks: int = 1
    num_time_masks: int = 1


def random_gain(waveform: torch.Tensor, min_gain: float = 0.7, max_gain: float = 1.3) -> torch.Tensor:
    if min_gain <= 0 or max_gain <= 0:
        raise ValueError("gain values must be positive")

    if min_gain > max_gain:
        raise ValueError("min_gain cannot be greater than max_gain")

    gain = random.uniform(min_gain, max_gain)
    return waveform * gain


def add_gaussian_noise(
    waveform: torch.Tensor,
    std_min: float = 0.001,
    std_max: float = 0.015,
) -> torch.Tensor:
    if std_min < 0 or std_max < 0:
        raise ValueError("noise std values cannot be negative")

    if std_min > std_max:
        raise ValueError("std_min cannot be greater than std_max")

    std = random.uniform(std_min, std_max)
    noise = torch.randn_like(waveform) * std
    return waveform + noise


def random_time_shift(waveform: torch.Tensor, max_shift_fraction: float = 0.1) -> torch.Tensor:
    """Circularly shift waveform along time dimension."""

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    if not 0 <= max_shift_fraction <= 1:
        raise ValueError("max_shift_fraction must be between 0 and 1")

    max_shift = int(waveform.shape[-1] * max_shift_fraction)

    if max_shift == 0:
        return waveform

    shift = random.randint(-max_shift, max_shift)
    return torch.roll(waveform, shifts=shift, dims=-1)


def augment_waveform(
    waveform: torch.Tensor,
    config: WaveformAugmentConfig | None = None,
) -> torch.Tensor:
    """Apply lightweight waveform augmentation.

    Used only for training. Never use this in validation/test unless you enjoy poisoning metrics.
    """

    cfg = config or WaveformAugmentConfig()

    if not cfg.enabled:
        return waveform

    augmented = waveform

    if random.random() < cfg.gain_prob:
        augmented = random_gain(augmented, cfg.min_gain, cfg.max_gain)

    if random.random() < cfg.noise_prob:
        augmented = add_gaussian_noise(augmented, cfg.noise_std_min, cfg.noise_std_max)

    if random.random() < cfg.shift_prob:
        augmented = random_time_shift(augmented, cfg.max_shift_fraction)

    return augmented.contiguous()


def frequency_mask(spec: torch.Tensor, max_width: int) -> torch.Tensor:
    """Mask random mel-frequency band.

    Supports:
        [mel, frames]
        [channels, mel, frames]
    """

    if spec.ndim not in {2, 3}:
        raise ValueError(f"Expected spec shape [mel, frames] or [channels, mel, frames], got {tuple(spec.shape)}")

    if max_width <= 0:
        return spec

    freq_dim = spec.shape[-2]

    if freq_dim <= 1:
        return spec

    width = random.randint(1, min(max_width, freq_dim))
    start = random.randint(0, freq_dim - width)

    masked = spec.clone()

    if masked.ndim == 2:
        masked[start : start + width, :] = 0.0
    else:
        masked[:, start : start + width, :] = 0.0

    return masked


def time_mask(spec: torch.Tensor, max_width: int) -> torch.Tensor:
    """Mask random time-frame band.

    Supports:
        [mel, frames]
        [channels, mel, frames]
    """

    if spec.ndim not in {2, 3}:
        raise ValueError(f"Expected spec shape [mel, frames] or [channels, mel, frames], got {tuple(spec.shape)}")

    if max_width <= 0:
        return spec

    time_dim = spec.shape[-1]

    if time_dim <= 1:
        return spec

    width = random.randint(1, min(max_width, time_dim))
    start = random.randint(0, time_dim - width)

    masked = spec.clone()
    masked[..., start : start + width] = 0.0

    return masked


def spec_augment(
    spec: torch.Tensor,
    config: SpecAugmentConfig | None = None,
) -> torch.Tensor:
    """Apply SpecAugment-style masking.

    Used for training only.
    """

    cfg = config or SpecAugmentConfig()

    if not cfg.enabled:
        return spec

    augmented = spec

    if random.random() < cfg.freq_mask_prob:
        for _ in range(cfg.num_freq_masks):
            augmented = frequency_mask(augmented, cfg.max_freq_mask_width)

    if random.random() < cfg.time_mask_prob:
        for _ in range(cfg.num_time_masks):
            augmented = time_mask(augmented, cfg.max_time_mask_width)

    return augmented.contiguous()