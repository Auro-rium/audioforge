from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForAudioClassification


@dataclass(frozen=True)
class ASTConfig:
    pretrained_name_or_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_labels: int = 200
    dropout: float = 0.1
    freeze_backbone: bool = False


class ASTAudioClassifier(nn.Module):
    """AST classifier for FSD50K multi-label audio classification.

    Expected input:
        [batch, max_length, num_mel_bins]

    Output:
        raw logits [batch, num_labels]

    Loss:
        BCEWithLogitsLoss
    """

    def __init__(self, config: ASTConfig | None = None) -> None:
        super().__init__()

        self.config = config or ASTConfig()

        hf_config = AutoConfig.from_pretrained(self.config.pretrained_name_or_path)
        hf_config.num_labels = self.config.num_labels
        hf_config.problem_type = "multi_label_classification"
        hf_config.hidden_dropout_prob = self.config.dropout

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.config.pretrained_name_or_path,
            config=hf_config,
            ignore_mismatched_sizes=True,
        )

        if self.config.freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for name, parameter in self.model.named_parameters():
            lower = name.lower()
            is_head = any(key in lower for key in ["classifier", "score", "head"])
            parameter.requires_grad = is_head

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.ndim != 3:
            raise ValueError(
                "ASTAudioClassifier expects [batch, max_length, num_mel_bins], "
                f"got {tuple(input_values.shape)}"
            )

        outputs = self.model(input_values=input_values)
        return outputs.logits


def create_ast_classifier(
    pretrained_name_or_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels: int = 200,
    dropout: float = 0.1,
    freeze_backbone: bool = False,
) -> ASTAudioClassifier:
    return ASTAudioClassifier(
        ASTConfig(
            pretrained_name_or_path=pretrained_name_or_path,
            num_labels=num_labels,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    )
