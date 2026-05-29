from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


ActivationName = Literal["relu", "gelu", "silu"]


@dataclass(frozen=True)
class ScratchCNNConfig:
    in_channels: int = 1
    num_labels: int = 200
    base_channels: int = 32
    dropout: float = 0.2
    activation: ActivationName = "gelu"
    use_batch_norm: bool = True


def get_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "gelu":
        return nn.GELU()

    if name == "silu":
        return nn.SiLU(inplace=True)

    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """Basic CNN block for log-mel spectrograms.

    Input shape:
        [batch, channels, n_mels, frames]

    Output shape:
        [batch, out_channels, reduced_n_mels, reduced_frames]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: ActivationName = "gelu",
        use_batch_norm: bool = True,
        pool: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not use_batch_norm,
            )
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(get_activation(activation))

        layers.append(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not use_batch_norm,
            )
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(get_activation(activation))

        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScratchAudioCNN(nn.Module):
    """From-scratch CNN baseline for audio spectrogram classification.

    This is intentionally simple, strong enough for a baseline, and not pretending
    to be a transformer wearing a Halloween costume.

    Expected input:
        logmel tensor with shape [batch, 1, n_mels, frames]

    Output:
        raw logits with shape [batch, num_labels]

    For FSD50K:
        num_labels = 200
        loss = BCEWithLogitsLoss
        prediction = sigmoid(logits)
    """

    def __init__(self, config: ScratchCNNConfig | None = None) -> None:
        super().__init__()

        self.config = config or ScratchCNNConfig()

        c = self.config.base_channels

        self.features = nn.Sequential(
            ConvBlock(
                self.config.in_channels,
                c,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                pool=True,
            ),
            ConvBlock(
                c,
                c * 2,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                pool=True,
            ),
            ConvBlock(
                c * 2,
                c * 4,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                pool=True,
            ),
            ConvBlock(
                c * 4,
                c * 8,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                pool=True,
            ),
            ConvBlock(
                c * 8,
                c * 8,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                pool=False,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(c * 8, c * 4),
            get_activation(self.config.activation),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(c * 4, self.config.num_labels),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ScratchAudioCNN expects input shape [batch, channels, n_mels, frames], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {x.shape[1]}"
            )

        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)

        return logits


def create_scratch_cnn(
    num_labels: int = 200,
    in_channels: int = 1,
    base_channels: int = 32,
    dropout: float = 0.2,
) -> ScratchAudioCNN:
    config = ScratchCNNConfig(
        in_channels=in_channels,
        num_labels=num_labels,
        base_channels=base_channels,
        dropout=dropout,
    )

    return ScratchAudioCNN(config)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return sum(parameter.numel() for parameter in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = 0

    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()

    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 * 1024)