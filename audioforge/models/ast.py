from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForAudioClassification

# This transformers version's ASTAttention uses q_proj/k_proj/v_proj/o_proj
# naming (verified directly against the loaded module tree, not the older
# query/key/value convention some ViT implementations use). q_proj/v_proj is
# the standard LoRA target for ViT-family encoders; leaving k_proj/o_proj
# untouched is the common LoRA-on-attention default, capturing most of the
# benefit at lower adapter parameter count.
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")


@dataclass(frozen=True)
class ASTConfig:
    pretrained_name_or_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_labels: int = 200
    dropout: float = 0.1

    # Full-backbone fine-tuning or full-backbone freeze (linear probe). Ignored
    # when use_lora=True: LoRA already keeps the backbone frozen by construction,
    # so combining it with a manual full freeze doesn't make sense.
    freeze_backbone: bool = False

    # PEFT (LoRA). This is the recommended way to adapt AST: AudioSet
    # pretraining already covers most of what FSD50K needs, so only a small
    # fraction of parameters (attention adapters + classifier head) need to
    # move, at a fraction of the memory/compute cost of full fine-tuning.
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LORA_TARGET_MODULES)

    def __post_init__(self) -> None:
        if self.use_lora and self.freeze_backbone:
            raise ValueError(
                "use_lora=True already freezes the backbone; freeze_backbone=True "
                "is redundant/conflicting with it. Set freeze_backbone=False."
            )


class ASTAudioClassifier(nn.Module):
    """AST classifier for FSD50K multi-label audio classification.

    Expected input:
        [batch, max_length, num_mel_bins]

    Output:
        raw logits [batch, num_labels]

    Loss:
        BCEWithLogitsLoss

    Training modes (see ASTConfig):
        use_lora=True (default): LoRA adapters on attention query/value
            projections, backbone otherwise frozen, classifier head fully
            trained. Recommended default.
        freeze_backbone=True: backbone fully frozen, only classifier head
            trains (linear probe).
        both False: full fine-tuning of every AST parameter.
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

        if self.config.use_lora:
            self._apply_lora()
        elif self.config.freeze_backbone:
            self.freeze_backbone()

    def _apply_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            modules_to_save=["classifier"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

    def freeze_backbone(self) -> None:
        for name, parameter in self.model.named_parameters():
            lower = name.lower()
            is_head = any(key in lower for key in ["classifier", "score", "head"])
            parameter.requires_grad = is_head

    def trainable_parameter_summary(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

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
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: tuple[str, ...] | list[str] = DEFAULT_LORA_TARGET_MODULES,
) -> ASTAudioClassifier:
    return ASTAudioClassifier(
        ASTConfig(
            pretrained_name_or_path=pretrained_name_or_path,
            num_labels=num_labels,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=tuple(lora_target_modules),
        )
    )
