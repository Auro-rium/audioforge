from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from audioforge.data.manifests import ManifestRow, read_manifest
from audioforge.evaluation.fsd50k_metrics import (
    FSD50KMetrics,
    compute_fsd50k_metrics,
    metrics_to_dict,
    save_metrics_json,
)
from audioforge.features.augment import (
    SpecAugmentConfig,
    WaveformAugmentConfig,
    augment_waveform,
    spec_augment,
)
from audioforge.features.logmel import LogMelExtractor, make_logmel_extractor
from audioforge.features.waveform import WaveformConfig, prepare_waveform
from audioforge.models.scratch_cnn import create_scratch_cnn
from audioforge.training.distributed import (
    create_accelerator,
    get_distributed_info,
    save_json,
    save_model_state,
    setup_torch_runtime,
)
from audioforge.utils.logging import configure_logging, get_logger
from audioforge.utils.seed import seed_everything

logger = get_logger(__name__)


@dataclass
class FSD50KTrainConfig:
    train_manifest: str = "data/manifests/fsd50k/train.csv"
    val_manifest: str = "data/manifests/fsd50k/val.csv"
    label_map_path: str = "data/manifests/fsd50k/label_map.json"

    output_dir: str = "outputs/fsd50k/scratch_cnn"
    checkpoint_dir: str = "outputs/fsd50k/scratch_cnn/checkpoints"
    resume_from: str | None = None

    model_name: str = "scratch_cnn"
    num_labels: int = 200
    base_channels: int = 32
    dropout: float = 0.2

    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    normalize_mode: str = "per_sample"

    epochs: int = 1
    batch_size: int = 8
    eval_batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0

    threshold: float = 0.5
    eval_every_steps: int = 500
    save_every_steps: int = 1000
    log_every_steps: int = 25

    max_train_samples: int | None = None
    max_val_samples: int | None = None

    seed: int = 42
    deterministic: bool = False
    log_level: str = "INFO"

    waveform_augment: bool = True
    spec_augment: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FSD50KTrainConfig:
        """Load config from flat or nested YAML.

        Accepts both:
            batch_size: 8

        and:
            training:
              batch_size: 8
        """

        data: dict[str, Any] = {}

        nested_keys = [
            "data",
            "model",
            "features",
            "training",
            "runtime",
            "augmentation",
        ]

        for key, value in raw.items():
            if key not in nested_keys:
                data[key] = value

        for section in nested_keys:
            value = raw.get(section)
            if isinstance(value, dict):
                data.update(value)

        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in data.items() if key in valid_fields}

        return cls(**filtered)


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_map: float = -1.0
    best_checkpoint: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_map = float(state.get("best_map", -1.0))
        self.best_checkpoint = state.get("best_checkpoint")


class FSD50KDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        label_map_path: str | Path,
        waveform_config: WaveformConfig,
        logmel_extractor: LogMelExtractor,
        training: bool,
        max_samples: int | None = None,
        waveform_augment_config: WaveformAugmentConfig | None = None,
        spec_augment_config: SpecAugmentConfig | None = None,
    ) -> None:
        self.rows = read_manifest(manifest_path)

        if max_samples is not None:
            self.rows = self.rows[:max_samples]

        self.label_to_id, self.id_to_label = load_label_info(label_map_path)
        self.num_labels = len(self.label_to_id)
        self.waveform_config = waveform_config
        self.logmel_extractor = logmel_extractor
        self.training = training
        self.waveform_augment_config = waveform_augment_config or WaveformAugmentConfig(
            enabled=training
        )
        self.spec_augment_config = spec_augment_config or SpecAugmentConfig(enabled=training)

        if not self.rows:
            raise ValueError(f"No rows loaded from manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]

        waveform = prepare_waveform(
            row.path,
            self.waveform_config,
            normalize_peak=True,
        )

        if self.training and self.waveform_augment_config.enabled:
            waveform = augment_waveform(waveform, self.waveform_augment_config)

        input_values = self.logmel_extractor(waveform)

        if self.training and self.spec_augment_config.enabled:
            input_values = spec_augment(input_values, self.spec_augment_config)

        labels = multi_hot_encode(row.labels, self.label_to_id)

        return {
            "input_values": input_values.float(),
            "labels": labels.float(),
            "path": row.path,
        }


def collate_fsd50k_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_values = torch.stack([item["input_values"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]

    return {
        "input_values": input_values,
        "labels": labels,
        "paths": paths,
    }


class FSD50KTrainer:
    def __init__(self, config: FSD50KTrainConfig) -> None:
        self.config = config

        configure_logging(level=config.log_level, json_logs=False)
        seed_everything(config.seed, deterministic=config.deterministic)
        setup_torch_runtime(deterministic=config.deterministic)

        self.accelerator = create_accelerator(
            output_dir=config.output_dir,
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        self.state = TrainState()
        self.accelerator.register_for_checkpointing(self.state)

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)

        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_json(self.output_dir / "train_config.json", asdict(config))
            save_json(self.output_dir / "distributed.json", get_distributed_info(self.accelerator))

    def train(self) -> None:
        model = self._build_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader, val_loader = self._build_dataloaders()

        total_update_steps = self._estimate_total_update_steps(train_loader)
        warmup_steps = int(total_update_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

        loss_fn = nn.BCEWithLogitsLoss()

        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )

        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        if self.accelerator.is_main_process:
            logger.info("Starting training")
            logger.info("Total update steps: %s", total_update_steps)
            logger.info("Warmup steps: %s", warmup_steps)

        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            self._train_one_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                epoch=epoch,
            )

            metrics = self.evaluate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                step=self.state.global_step,
                epoch=epoch,
            )

            self._save_checkpoint(
                name=f"epoch_{epoch + 1}",
                model=model,
                metrics=metrics,
            )

            self.state.epoch = epoch + 1

        if self.accelerator.is_main_process:
            logger.info("Training complete")
            logger.info("Best mAP: %.6f", self.state.best_map)
            logger.info("Best checkpoint: %s", self.state.best_checkpoint)

    def _train_one_epoch(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int,
    ) -> None:
        model.train()

        running_loss = 0.0
        running_count = 0
        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(train_loader):
            with self.accelerator.accumulate(model):
                logits = model(batch["input_values"])
                loss = loss_fn(logits, batch["labels"])

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            reduced_loss = self.accelerator.reduce(loss.detach(), reduction="mean")
            running_loss += float(reduced_loss.item())
            running_count += 1

            if self.accelerator.sync_gradients:
                self.state.global_step += 1

                if self.state.global_step % self.config.log_every_steps == 0:
                    avg_loss = running_loss / max(running_count, 1)
                    lr = scheduler.get_last_lr()[0]

                    if self.accelerator.is_main_process:
                        logger.info(
                            "epoch=%s step=%s batch=%s loss=%.6f lr=%.8f",
                            epoch + 1,
                            self.state.global_step,
                            batch_index,
                            avg_loss,
                            lr,
                        )

                    running_loss = 0.0
                    running_count = 0

                if self.state.global_step % self.config.eval_every_steps == 0:
                    self.evaluate(
                        model=model,
                        val_loader=val_loader,
                        loss_fn=loss_fn,
                        step=self.state.global_step,
                        epoch=epoch,
                    )
                    model.train()

                if self.state.global_step % self.config.save_every_steps == 0:
                    self._save_checkpoint(
                        name=f"step_{self.state.global_step}",
                        model=model,
                        metrics=None,
                    )

        elapsed = time.perf_counter() - epoch_start

        if self.accelerator.is_main_process:
            logger.info("Finished epoch=%s elapsed_seconds=%.2f", epoch + 1, elapsed)

    @torch.no_grad()
    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        step: int,
        epoch: int,
    ) -> FSD50KMetrics:
        model.eval()

        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in val_loader:
            logits = model(batch["input_values"])
            loss = loss_fn(logits, batch["labels"])

            reduced_loss = self.accelerator.reduce(loss.detach(), reduction="mean")
            losses.append(float(reduced_loss.item()))

            gathered_logits = self.accelerator.gather_for_metrics(logits.detach())
            gathered_labels = self.accelerator.gather_for_metrics(batch["labels"].detach())

            all_logits.append(gathered_logits.cpu())
            all_labels.append(gathered_labels.cpu())

        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        _, id_to_label = load_label_info(self.config.label_map_path)
        label_names = [id_to_label[index] for index in range(len(id_to_label))]

        metrics = compute_fsd50k_metrics(
            logits_tensor,
            labels_tensor,
            from_logits=True,
            threshold=self.config.threshold,
            label_names=label_names,
        )

        mean_loss = sum(losses) / max(len(losses), 1)

        if self.accelerator.is_main_process:
            logger.info(
                "eval epoch=%s step=%s loss=%.6f mAP=%.6f micro_f1=%.6f macro_f1=%.6f",
                epoch + 1,
                step,
                mean_loss,
                metrics.mAP,
                metrics.micro_f1,
                metrics.macro_f1,
            )

            metrics_path = self.output_dir / "metrics" / f"eval_step_{step}.json"
            save_metrics_json(metrics, metrics_path, include_per_class=True)

        if metrics.mAP > self.state.best_map:
            self.state.best_map = metrics.mAP
            self._save_best_model(model=model, metrics=metrics, step=step, epoch=epoch)

        self.accelerator.wait_for_everyone()
        return metrics

    def _build_model(self) -> torch.nn.Module:
        if self.config.model_name != "scratch_cnn":
            raise ValueError(
                f"Unsupported model_name={self.config.model_name}. "
                "Action 6 currently supports scratch_cnn only."
            )

        return create_scratch_cnn(
            num_labels=self.config.num_labels,
            in_channels=1,
            base_channels=self.config.base_channels,
            dropout=self.config.dropout,
        )

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        waveform_config = WaveformConfig(
            sample_rate=self.config.sample_rate,
            clip_seconds=self.config.clip_seconds,
            mono=True,
            crop_mode="random",
        )

        val_waveform_config = WaveformConfig(
            sample_rate=self.config.sample_rate,
            clip_seconds=self.config.clip_seconds,
            mono=True,
            crop_mode="center",
        )

        train_logmel = make_logmel_extractor(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            normalize_mode=self.config.normalize_mode,  # type: ignore[arg-type]
        )

        val_logmel = make_logmel_extractor(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            normalize_mode=self.config.normalize_mode,  # type: ignore[arg-type]
        )

        train_dataset = FSD50KDataset(
            manifest_path=self.config.train_manifest,
            label_map_path=self.config.label_map_path,
            waveform_config=waveform_config,
            logmel_extractor=train_logmel,
            training=True,
            max_samples=self.config.max_train_samples,
            waveform_augment_config=WaveformAugmentConfig(enabled=self.config.waveform_augment),
            spec_augment_config=SpecAugmentConfig(enabled=self.config.spec_augment),
        )

        val_dataset = FSD50KDataset(
            manifest_path=self.config.val_manifest,
            label_map_path=self.config.label_map_path,
            waveform_config=val_waveform_config,
            logmel_extractor=val_logmel,
            training=False,
            max_samples=self.config.max_val_samples,
            waveform_augment_config=WaveformAugmentConfig(enabled=False),
            spec_augment_config=SpecAugmentConfig(enabled=False),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=collate_fsd50k_batch,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=collate_fsd50k_batch,
            drop_last=False,
        )

        return train_loader, val_loader

    def _estimate_total_update_steps(self, train_loader: DataLoader) -> int:
        micro_steps_per_epoch = len(train_loader)
        update_steps_per_epoch = math.ceil(
            micro_steps_per_epoch / self.config.gradient_accumulation_steps
        )
        return max(update_steps_per_epoch * self.config.epochs, 1)

    def _save_checkpoint(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        metrics: FSD50KMetrics | None,
    ) -> None:
        checkpoint_path = self.checkpoint_dir / name

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(checkpoint_path))

        if self.accelerator.is_main_process:
            payload = {
                "name": name,
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_map": self.state.best_map,
                "best_checkpoint": self.state.best_checkpoint,
                "metrics": metrics_to_dict(metrics, include_per_class=False) if metrics else None,
            }
            save_json(checkpoint_path / "trainer_state.json", payload)
            logger.info("Saved checkpoint: %s", checkpoint_path)

    def _save_best_model(
        self,
        *,
        model: torch.nn.Module,
        metrics: FSD50KMetrics,
        step: int,
        epoch: int,
    ) -> None:
        best_path = self.output_dir / "best" / "scratch_cnn_best.pt"

        extra = {
            "epoch": epoch,
            "global_step": step,
            "metrics": metrics_to_dict(metrics, include_per_class=False),
            "config": asdict(self.config),
        }

        save_model_state(
            accelerator=self.accelerator,
            model=model,
            output_path=best_path,
            extra=extra,
        )

        if self.accelerator.is_main_process:
            self.state.best_checkpoint = str(best_path)
            save_metrics_json(
                metrics,
                self.output_dir / "best" / "best_metrics.json",
                include_per_class=True,
            )
            logger.info("Saved new best model: %s mAP=%.6f", best_path, metrics.mAP)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        self.accelerator.load_state(str(checkpoint))

        if self.accelerator.is_main_process:
            logger.info("Loaded checkpoint: %s", checkpoint)


def load_train_config(path: str | Path | None = None) -> FSD50KTrainConfig:
    if path is None:
        return FSD50KTrainConfig()

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")

    return FSD50KTrainConfig.from_dict(raw)


def load_label_info(label_map_path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    path = Path(label_map_path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    label_to_id_raw = payload.get("label_to_id")
    id_to_label_raw = payload.get("id_to_label")

    if not isinstance(label_to_id_raw, dict):
        raise ValueError(f"Invalid label_map.json missing label_to_id: {path}")

    label_to_id = {str(label): int(idx) for label, idx in label_to_id_raw.items()}

    if isinstance(id_to_label_raw, dict):
        id_to_label = {int(idx): str(label) for idx, label in id_to_label_raw.items()}
    else:
        id_to_label = {idx: label for label, idx in label_to_id.items()}

    return label_to_id, id_to_label


def multi_hot_encode(labels: list[str], label_to_id: dict[str, int]) -> torch.Tensor:
    output = torch.zeros(len(label_to_id), dtype=torch.float32)

    for label in labels:
        if label not in label_to_id:
            raise KeyError(f"Label not found in label map: {label}")
        output[label_to_id[label]] = 1.0

    return output