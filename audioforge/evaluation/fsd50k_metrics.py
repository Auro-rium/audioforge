from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


@dataclass(frozen=True)
class PerClassMetrics:
    label_id: int
    label_name: str | None
    average_precision: float | None
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class FSD50KMetrics:
    mAP: float
    micro_average_precision: float
    macro_f1: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    micro_precision: float
    micro_recall: float
    threshold: float
    num_samples: int
    num_labels: int
    per_class: list[PerClassMetrics]


def _to_numpy(value: torch.Tensor | np.ndarray | list[Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return np.asarray(value)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _validate_multilabel_arrays(
    scores: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [num_samples, num_labels], got {scores.shape}")

    if targets.ndim != 2:
        raise ValueError(f"targets must have shape [num_samples, num_labels], got {targets.shape}")

    if scores.shape != targets.shape:
        raise ValueError(f"scores and targets shape mismatch: {scores.shape} vs {targets.shape}")

    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")

    if not np.isfinite(targets).all():
        raise ValueError("targets contain NaN or Inf")

    # Strict enough to catch broken labels, tolerant enough for float tensors.
    if not np.isin(targets, [0.0, 1.0]).all():
        raise ValueError("targets must be binary multi-hot values containing only 0/1")

    targets = (targets > 0.5).astype(np.int32)

    return scores, targets


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return AP or None when AP is undefined.

    A class with no positive examples has undefined AP. We do not fake it as 0.
    Fake numbers are how dashboards become decorative lies.
    """

    if y_true.sum() == 0:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        value = average_precision_score(y_true, y_score)

    if not np.isfinite(value):
        return None

    return float(value)


def compute_per_class_average_precision(
    scores: torch.Tensor | np.ndarray | list[Any],
    targets: torch.Tensor | np.ndarray | list[Any],
    *,
    label_names: list[str] | None = None,
) -> list[PerClassMetrics]:
    scores_np, targets_np = _validate_multilabel_arrays(
        _to_numpy(scores),
        _to_numpy(targets),
    )

    num_labels = targets_np.shape[1]

    if label_names is not None and len(label_names) != num_labels:
        raise ValueError(f"label_names length {len(label_names)} != num_labels {num_labels}")

    per_class: list[PerClassMetrics] = []

    predicted = np.zeros_like(targets_np)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np,
        predicted,
        average=None,
        zero_division=0,
    )

    for label_id in range(num_labels):
        ap = _safe_average_precision(targets_np[:, label_id], scores_np[:, label_id])

        per_class.append(
            PerClassMetrics(
                label_id=label_id,
                label_name=label_names[label_id] if label_names is not None else None,
                average_precision=ap,
                precision=float(precision[label_id]),
                recall=float(recall[label_id]),
                f1=float(f1[label_id]),
                support=int(support[label_id]),
            )
        )

    return per_class


def compute_fsd50k_metrics(
    predictions: torch.Tensor | np.ndarray | list[Any],
    targets: torch.Tensor | np.ndarray | list[Any],
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
    label_names: list[str] | None = None,
) -> FSD50KMetrics:
    """Compute FSD50K multi-label metrics.

    Args:
        predictions:
            Raw logits if from_logits=True, otherwise probabilities.
            Shape [num_samples, num_labels].

        targets:
            Multi-hot binary targets.
            Shape [num_samples, num_labels].

        from_logits:
            Whether to apply sigmoid to predictions.

        threshold:
            Probability threshold for precision/recall/F1 metrics.

        label_names:
            Optional label names aligned to label ids.

    Returns:
        FSD50KMetrics dataclass.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    pred_np = _to_numpy(predictions).astype(np.float32)

    if from_logits:
        scores_np = sigmoid_numpy(pred_np)
    else:
        scores_np = pred_np

    scores_np, targets_np = _validate_multilabel_arrays(
        scores_np,
        _to_numpy(targets),
    )

    if scores_np.min() < 0.0 or scores_np.max() > 1.0:
        raise ValueError(
            "Probability scores must be in [0, 1]. "
            "If passing logits, use from_logits=True."
        )

    num_samples, num_labels = targets_np.shape

    if label_names is not None and len(label_names) != num_labels:
        raise ValueError(f"label_names length {len(label_names)} != num_labels {num_labels}")

    binary_predictions = (scores_np >= threshold).astype(np.int32)

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            targets_np,
            binary_predictions,
            average=None,
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets_np,
        binary_predictions,
        average="macro",
        zero_division=0,
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        targets_np,
        binary_predictions,
        average="micro",
        zero_division=0,
    )

    per_class: list[PerClassMetrics] = []
    per_class_ap_values: list[float] = []

    for label_id in range(num_labels):
        ap = _safe_average_precision(targets_np[:, label_id], scores_np[:, label_id])

        if ap is not None:
            per_class_ap_values.append(ap)

        per_class.append(
            PerClassMetrics(
                label_id=label_id,
                label_name=label_names[label_id] if label_names is not None else None,
                average_precision=ap,
                precision=float(per_class_precision[label_id]),
                recall=float(per_class_recall[label_id]),
                f1=float(per_class_f1[label_id]),
                support=int(per_class_support[label_id]),
            )
        )

    mAP = float(np.mean(per_class_ap_values)) if per_class_ap_values else 0.0

    micro_ap = _safe_average_precision(
        targets_np.reshape(-1),
        scores_np.reshape(-1),
    )

    return FSD50KMetrics(
        mAP=mAP,
        micro_average_precision=float(micro_ap) if micro_ap is not None else 0.0,
        macro_f1=float(macro_f1),
        micro_f1=float(micro_f1),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        micro_precision=float(micro_precision),
        micro_recall=float(micro_recall),
        threshold=float(threshold),
        num_samples=int(num_samples),
        num_labels=int(num_labels),
        per_class=per_class,
    )


def find_best_global_threshold(
    predictions: torch.Tensor | np.ndarray | list[Any],
    targets: torch.Tensor | np.ndarray | list[Any],
    *,
    from_logits: bool = True,
    metric: str = "micro_f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, FSD50KMetrics]:
    """Find a single global threshold on validation data.

    Use this only on validation, never test. Test-set threshold tuning is benchmark sin.
    """

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    if metric not in {"micro_f1", "macro_f1", "micro_precision", "micro_recall"}:
        raise ValueError(f"Unsupported threshold search metric: {metric}")

    best_threshold = float(thresholds[0])
    best_metrics: FSD50KMetrics | None = None
    best_value = -math.inf

    for threshold in thresholds:
        current = compute_fsd50k_metrics(
            predictions,
            targets,
            from_logits=from_logits,
            threshold=float(threshold),
        )

        value = float(getattr(current, metric))

        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = current

    if best_metrics is None:
        raise RuntimeError("Threshold search failed")

    return best_threshold, best_metrics


def metrics_to_dict(metrics: FSD50KMetrics, *, include_per_class: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mAP": metrics.mAP,
        "micro_average_precision": metrics.micro_average_precision,
        "macro_f1": metrics.macro_f1,
        "micro_f1": metrics.micro_f1,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "micro_precision": metrics.micro_precision,
        "micro_recall": metrics.micro_recall,
        "threshold": metrics.threshold,
        "num_samples": metrics.num_samples,
        "num_labels": metrics.num_labels,
    }

    if include_per_class:
        payload["per_class"] = [asdict(item) for item in metrics.per_class]

    return _json_safe(payload)


def save_metrics_json(
    metrics: FSD50KMetrics,
    path: str | Path,
    *,
    include_per_class: bool = True,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = metrics_to_dict(metrics, include_per_class=include_per_class)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf to None so JSON stays clean."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value