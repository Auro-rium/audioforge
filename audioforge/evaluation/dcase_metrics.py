from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class DCASEPrediction:
    path: str
    machine_type: str
    section: str
    domain: str
    y_true: int
    anomaly_score: float


@dataclass(frozen=True)
class DCASEGroupMetrics:
    group: str
    count: int
    positives: int
    negatives: int
    auc: float | None
    pauc: float | None
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class DCASEMetrics:
    official_score: float
    mean_auc: float
    mean_pauc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    pauc_max_fpr: float
    num_samples: int
    num_positive: int
    num_negative: int
    groups: list[DCASEGroupMetrics]


PREDICTION_COLUMNS = [
    "path",
    "machine_type",
    "section",
    "domain",
    "y_true",
    "anomaly_score",
]


def harmonic_mean(values: Iterable[float], *, eps: float = 1e-12) -> float:
    clean = [float(value) for value in values if np.isfinite(value) and value > 0]

    if not clean:
        return 0.0

    return len(clean) / sum(1.0 / max(value, eps) for value in clean)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    value = roc_auc_score(y_true, scores)

    if not np.isfinite(value):
        return None

    return float(value)


def safe_pauc(y_true: np.ndarray, scores: np.ndarray, *, max_fpr: float = 0.1) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    value = roc_auc_score(y_true, scores, max_fpr=max_fpr)

    if not np.isfinite(value):
        return None

    return float(value)


def read_dcase_predictions(path: str | Path) -> list[DCASEPrediction]:
    prediction_path = Path(path)

    if not prediction_path.exists():
        raise FileNotFoundError(f"DCASE prediction CSV not found: {prediction_path}")

    rows: list[DCASEPrediction] = []

    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        missing = set(PREDICTION_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

        for raw in reader:
            rows.append(
                DCASEPrediction(
                    path=raw["path"],
                    machine_type=raw["machine_type"],
                    section=raw["section"],
                    domain=raw["domain"],
                    y_true=int(raw["y_true"]),
                    anomaly_score=float(raw["anomaly_score"]),
                )
            )

    return rows


def write_dcase_predictions(path: str | Path, rows: Iterable[DCASEPrediction]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "machine_type": row.machine_type,
                    "section": row.section,
                    "domain": row.domain,
                    "y_true": row.y_true,
                    "anomaly_score": f"{row.anomaly_score:.10f}",
                }
            )


def compute_dcase_metrics(
    predictions: list[DCASEPrediction],
    *,
    threshold: float | None = None,
    pauc_max_fpr: float = 0.1,
    group_by: tuple[str, ...] = ("machine_type", "section", "domain"),
) -> DCASEMetrics:
    """Compute DCASE-style anomaly metrics.

    y_true:
        0 = normal
        1 = anomaly

    anomaly_score:
        larger means more anomalous

    official_score:
        harmonic mean over group-level AUC and pAUC values.
        This is official-style, not a replacement for the released evaluator script.
    """

    if not predictions:
        raise ValueError("No DCASE predictions provided")

    y_true = np.asarray([row.y_true for row in predictions], dtype=np.int32)
    scores = np.asarray([row.anomaly_score for row in predictions], dtype=np.float64)

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1")

    if not np.isfinite(scores).all():
        raise ValueError("anomaly_score contains NaN or Inf")

    if threshold is None:
        threshold = float(np.median(scores))

    y_pred = (scores >= threshold).astype(np.int32)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    grouped: dict[str, list[DCASEPrediction]] = {}

    for row in predictions:
        key_parts: list[str] = []

        for key in group_by:
            value = getattr(row, key)
            key_parts.append(str(value))

        group_key = "/".join(key_parts)
        grouped.setdefault(group_key, []).append(row)

    group_metrics: list[DCASEGroupMetrics] = []
    auc_values: list[float] = []
    pauc_values: list[float] = []

    for group_key, group_rows in sorted(grouped.items()):
        group_y_true = np.asarray([row.y_true for row in group_rows], dtype=np.int32)
        group_scores = np.asarray([row.anomaly_score for row in group_rows], dtype=np.float64)
        group_pred = (group_scores >= threshold).astype(np.int32)

        auc = safe_auc(group_y_true, group_scores)
        pauc = safe_pauc(group_y_true, group_scores, max_fpr=pauc_max_fpr)

        if auc is not None:
            auc_values.append(auc)

        if pauc is not None:
            pauc_values.append(pauc)

        group_metrics.append(
            DCASEGroupMetrics(
                group=group_key,
                count=len(group_rows),
                positives=int(group_y_true.sum()),
                negatives=int(len(group_y_true) - group_y_true.sum()),
                auc=auc,
                pauc=pauc,
                precision=float(precision_score(group_y_true, group_pred, zero_division=0)),
                recall=float(recall_score(group_y_true, group_pred, zero_division=0)),
                f1=float(f1_score(group_y_true, group_pred, zero_division=0)),
            )
        )

    mean_auc = float(np.mean(auc_values)) if auc_values else 0.0
    mean_pauc = float(np.mean(pauc_values)) if pauc_values else 0.0

    # DCASE-style ranking focuses on harmonic behavior: bad AUC/pAUC should hurt.
    official_score = harmonic_mean([*auc_values, *pauc_values])

    return DCASEMetrics(
        official_score=official_score,
        mean_auc=mean_auc,
        mean_pauc=mean_pauc,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=float(threshold),
        pauc_max_fpr=float(pauc_max_fpr),
        num_samples=int(len(predictions)),
        num_positive=int(y_true.sum()),
        num_negative=int(len(y_true) - y_true.sum()),
        groups=group_metrics,
    )


def metrics_to_dict(metrics: DCASEMetrics, *, include_groups: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "official_score": metrics.official_score,
        "mean_auc": metrics.mean_auc,
        "mean_pauc": metrics.mean_pauc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "threshold": metrics.threshold,
        "pauc_max_fpr": metrics.pauc_max_fpr,
        "num_samples": metrics.num_samples,
        "num_positive": metrics.num_positive,
        "num_negative": metrics.num_negative,
    }

    if include_groups:
        payload["groups"] = [asdict(group) for group in metrics.groups]

    return _json_safe(payload)


def save_dcase_metrics_json(
    metrics: DCASEMetrics,
    path: str | Path,
    *,
    include_groups: bool = True,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = metrics_to_dict(metrics, include_groups=include_groups)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value
