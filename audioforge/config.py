from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeConfig(BaseModel):
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class DataConfig(BaseModel):
    dataset_name: str = "fsd50k"
    data_root: Path = Path("data")
    manifest_dir: Path = Path("data/manifests")
    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    num_workers: int = 4
    pin_memory: bool = True

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return value

    @field_validator("clip_seconds")
    @classmethod
    def validate_clip_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("clip_seconds must be positive")
        return value


class FeatureConfig(BaseModel):
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    normalize: bool = True


class ModelConfig(BaseModel):
    name: str = "scratch_cnn"
    pretrained_name_or_path: str | None = None
    num_labels: int = 200
    dropout: float = 0.1
    freeze_backbone: bool = False


class TrainingConfig(BaseModel):
    task: Literal["fsd50k", "dcase"] = "fsd50k"
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    save_every_steps: int = 500
    eval_every_steps: int = 500
    max_grad_norm: float = 1.0

    @field_validator("epochs", "batch_size", "gradient_accumulation_steps")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm")
    @classmethod
    def validate_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value cannot be negative")
        return value


class MetricsConfig(BaseModel):
    prometheus_enabled: bool = False
    prometheus_host: str = "0.0.0.0"
    prometheus_port: int = 9090


class AudioForgeConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class EnvSettings(BaseSettings):
    """Environment overrides.

    Example:
        AUDIOFORGE_LOG_LEVEL=DEBUG
        AUDIOFORGE_PROMETHEUS_ENABLED=true
    """

    model_config = SettingsConfigDict(
        env_prefix="AUDIOFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str | None = None
    prometheus_enabled: bool | None = None
    prometheus_port: int | None = None


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return raw


def apply_env_overrides(config: AudioForgeConfig) -> AudioForgeConfig:
    env = EnvSettings()

    data = config.model_dump()

    if env.log_level is not None:
        data["runtime"]["log_level"] = env.log_level.upper()

    if env.prometheus_enabled is not None:
        data["metrics"]["prometheus_enabled"] = env.prometheus_enabled

    if env.prometheus_port is not None:
        data["metrics"]["prometheus_port"] = env.prometheus_port

    return AudioForgeConfig.model_validate(data)


def load_config(path: str | Path | None = None) -> AudioForgeConfig:
    """Load AudioForge config from defaults + optional YAML + environment overrides."""

    base = AudioForgeConfig().model_dump()

    if path is not None:
        yaml_data = load_yaml_config(path)
        base = _deep_update(base, yaml_data)

    config = AudioForgeConfig.model_validate(base)
    return apply_env_overrides(config)