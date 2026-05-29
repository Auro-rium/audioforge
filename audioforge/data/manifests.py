from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = ["path", "split", "labels", "duration"]


@dataclass(frozen=True)
class ManifestRow:
    path: str
    split: str
    labels: list[str]
    duration: float


@dataclass(frozen=True)
class ManifestStats:
    split: str
    rows: int
    missing_audio: int
    total_duration_hours: float
    unique_labels: int


def parse_label_string(value: str) -> list[str]:
    """Parse FSD50K comma-separated labels.

    FSD50K labels are clip-level weak labels. One clip can have many labels.
    """

    if value is None:
        return []

    return [label.strip() for label in str(value).split(",") if label.strip()]


def labels_to_manifest_string(labels: Iterable[str]) -> str:
    return ",".join(label.strip() for label in labels if label.strip())


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[ManifestRow] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        missing_columns = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Manifest {manifest_path} is missing columns: {sorted(missing_columns)}"
            )

        for raw in reader:
            rows.append(
                ManifestRow(
                    path=raw["path"],
                    split=raw["split"],
                    labels=parse_label_string(raw["labels"]),
                    duration=float(raw["duration"]),
                )
            )

    return rows


def write_manifest(path: str | Path, rows: Iterable[ManifestRow]) -> ManifestStats:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    row_list = list(rows)
    unique_labels = sorted({label for row in row_list for label in row.labels})
    total_seconds = sum(row.duration for row in row_list)

    split = row_list[0].split if row_list else "unknown"

    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()

        for row in row_list:
            writer.writerow(
                {
                    "path": row.path,
                    "split": row.split,
                    "labels": labels_to_manifest_string(row.labels),
                    "duration": f"{row.duration:.6f}",
                }
            )

    return ManifestStats(
        split=split,
        rows=len(row_list),
        missing_audio=0,
        total_duration_hours=round(total_seconds / 3600.0, 4),
        unique_labels=len(unique_labels),
    )


def write_label_map(
    path: str | Path,
    labels: list[str],
    *,
    dataset_name: str = "fsd50k",
) -> None:
    """Write stable label map for multi-label classification."""

    label_map_path = Path(path)
    label_map_path.parent.mkdir(parents=True, exist_ok=True)

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}

    payload = {
        "dataset": dataset_name,
        "num_labels": len(labels),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }

    with label_map_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_manifest_summary(path: str | Path, stats: list[ManifestStats]) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "manifests": [asdict(item) for item in stats],
        "total_rows": sum(item.rows for item in stats),
        "total_duration_hours": round(sum(item.total_duration_hours for item in stats), 4),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)