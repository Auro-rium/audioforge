from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from audioforge.data.manifests import (
    ManifestRow,
    ManifestStats,
    parse_label_string,
    write_label_map,
    write_manifest,
    write_manifest_summary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FSD50KPaths:
    root: Path
    dev_audio_dir: Path
    eval_audio_dir: Path
    ground_truth_dir: Path
    dev_csv: Path
    eval_csv: Path
    vocabulary_csv: Path

    @classmethod
    def from_root(cls, root: str | Path) -> FSD50KPaths:
        root_path = Path(root)

        return cls(
            root=root_path,
            dev_audio_dir=root_path / "FSD50K.dev_audio",
            eval_audio_dir=root_path / "FSD50K.eval_audio",
            ground_truth_dir=root_path / "FSD50K.ground_truth",
            dev_csv=root_path / "FSD50K.ground_truth" / "dev.csv",
            eval_csv=root_path / "FSD50K.ground_truth" / "eval.csv",
            vocabulary_csv=root_path / "FSD50K.ground_truth" / "vocabulary.csv",
        )


@dataclass(frozen=True)
class FSD50KBuildResult:
    output_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    label_map_json: Path
    summary_json: Path
    stats: list[ManifestStats]


def validate_fsd50k_root(paths: FSD50KPaths) -> None:
    required = [
        paths.root,
        paths.dev_audio_dir,
        paths.eval_audio_dir,
        paths.ground_truth_dir,
        paths.dev_csv,
        paths.eval_csv,
        paths.vocabulary_csv,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        joined = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "FSD50K root is incomplete. Missing required files/directories:\n"
            f"{joined}\n\n"
            "Expected layout:\n"
            "data/raw/fsd50k/\n"
            "  FSD50K.dev_audio/\n"
            "  FSD50K.eval_audio/\n"
            "  FSD50K.ground_truth/dev.csv\n"
            "  FSD50K.ground_truth/eval.csv\n"
            "  FSD50K.ground_truth/vocabulary.csv"
        )


def read_vocabulary(vocabulary_csv: str | Path) -> list[str]:
    """Read FSD50K vocabulary in stable order.

    Official vocabulary rows are usually CSV rows containing an index, FSD50K label,
    and AudioSet MID. This parser is intentionally tolerant because dataset mirrors
    sometimes preserve headers differently, because naturally even CSV cannot behave.
    """

    path = Path(vocabulary_csv)

    if not path.exists():
        raise FileNotFoundError(f"Vocabulary CSV not found: {path}")

    labels: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            first_cell = row[0].strip().lower()

            if first_cell in {"index", "idx", "id"}:
                continue

            if len(row) >= 2:
                label = row[1].strip()
            else:
                label = row[0].strip()

            if label and label not in labels:
                labels.append(label)

    if not labels:
        raise ValueError(f"No labels found in vocabulary: {path}")

    return labels


def _safe_audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0:
            return 0.0
        return float(info.frames) / float(info.samplerate)
    except Exception as exc:
        logger.warning("Failed to read audio duration for %s: %s", path, exc)
        return 0.0


def _as_output_path(path: Path, *, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())

    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _read_ground_truth_rows(
    csv_path: Path,
    *,
    audio_dir: Path,
    default_split: str,
    absolute_paths: bool,
    fail_on_missing_audio: bool,
) -> tuple[list[ManifestRow], int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")

    rows: list[ManifestRow] = []
    missing_audio = 0

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"fname", "labels"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path} missing required columns: {sorted(missing_columns)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for raw in reader:
            fname = str(raw["fname"]).strip()

            if not fname:
                continue

            audio_path = audio_dir / f"{fname}.wav"

            if not audio_path.exists():
                missing_audio += 1

                if fail_on_missing_audio:
                    raise FileNotFoundError(f"Audio file missing: {audio_path}")

                continue

            split = str(raw.get("split") or default_split).strip().lower()
            labels = parse_label_string(raw["labels"])
            duration = _safe_audio_duration(audio_path)

            rows.append(
                ManifestRow(
                    path=_as_output_path(audio_path, absolute_paths=absolute_paths),
                    split=split,
                    labels=labels,
                    duration=duration,
                )
            )

    return rows, missing_audio


def build_fsd50k_manifests(
    root: str | Path,
    output_dir: str | Path,
    *,
    absolute_paths: bool = False,
    fail_on_missing_audio: bool = False,
) -> FSD50KBuildResult:
    """Build train/val/test manifests and label_map.json for FSD50K.

    Official split behavior:
    - dev.csv contains train/val in the `split` column.
    - eval.csv becomes our `test.csv`.
    """

    paths = FSD50KPaths.from_root(root)
    validate_fsd50k_root(paths)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = read_vocabulary(paths.vocabulary_csv)

    dev_rows, dev_missing = _read_ground_truth_rows(
        paths.dev_csv,
        audio_dir=paths.dev_audio_dir,
        default_split="train",
        absolute_paths=absolute_paths,
        fail_on_missing_audio=fail_on_missing_audio,
    )

    eval_rows, eval_missing = _read_ground_truth_rows(
        paths.eval_csv,
        audio_dir=paths.eval_audio_dir,
        default_split="test",
        absolute_paths=absolute_paths,
        fail_on_missing_audio=fail_on_missing_audio,
    )

    train_rows = [row for row in dev_rows if row.split == "train"]
    val_rows = [row for row in dev_rows if row.split in {"val", "valid", "validation"}]

    normalized_val_rows = [
        ManifestRow(
            path=row.path,
            split="val",
            labels=row.labels,
            duration=row.duration,
        )
        for row in val_rows
    ]

    test_rows = [
        ManifestRow(
            path=row.path,
            split="test",
            labels=row.labels,
            duration=row.duration,
        )
        for row in eval_rows
    ]

    if not train_rows:
        raise ValueError("No train rows found. Check FSD50K.ground_truth/dev.csv split column.")

    if not normalized_val_rows:
        raise ValueError("No val rows found. Check FSD50K.ground_truth/dev.csv split column.")

    if not test_rows:
        raise ValueError("No test rows found. Check FSD50K.ground_truth/eval.csv.")

    train_csv = output_path / "train.csv"
    val_csv = output_path / "val.csv"
    test_csv = output_path / "test.csv"
    label_map_json = output_path / "label_map.json"
    summary_json = output_path / "summary.json"

    stats = [
        write_manifest(train_csv, train_rows),
        write_manifest(val_csv, normalized_val_rows),
        write_manifest(test_csv, test_rows),
    ]

    stats = [
        ManifestStats(
            split=stats[0].split,
            rows=stats[0].rows,
            missing_audio=dev_missing,
            total_duration_hours=stats[0].total_duration_hours,
            unique_labels=stats[0].unique_labels,
        ),
        ManifestStats(
            split=stats[1].split,
            rows=stats[1].rows,
            missing_audio=0,
            total_duration_hours=stats[1].total_duration_hours,
            unique_labels=stats[1].unique_labels,
        ),
        ManifestStats(
            split=stats[2].split,
            rows=stats[2].rows,
            missing_audio=eval_missing,
            total_duration_hours=stats[2].total_duration_hours,
            unique_labels=stats[2].unique_labels,
        ),
    ]

    write_label_map(label_map_json, labels, dataset_name="fsd50k")
    write_manifest_summary(summary_json, stats)

    _write_human_summary(output_path / "README.manifest.md", stats, label_map_json)

    return FSD50KBuildResult(
        output_dir=output_path,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        label_map_json=label_map_json,
        summary_json=summary_json,
        stats=stats,
    )


def _write_human_summary(path: Path, stats: list[ManifestStats], label_map_json: Path) -> None:
    lines = [
        "# FSD50K Manifest Summary",
        "",
        "| Split | Rows | Missing Audio | Duration Hours | Unique Labels |",
        "|---|---:|---:|---:|---:|",
    ]

    for item in stats:
        lines.append(
            f"| {item.split} | {item.rows} | {item.missing_audio} | "
            f"{item.total_duration_hours:.4f} | {item.unique_labels} |"
        )

    lines.extend(
        [
            "",
            f"Label map: `{label_map_json.name}`",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def load_label_map(path: str | Path) -> dict[str, int]:
    label_map_path = Path(path)

    with label_map_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    label_to_id = payload.get("label_to_id")
    if not isinstance(label_to_id, dict):
        raise ValueError(f"Invalid label map: {label_map_path}")

    return {str(label): int(idx) for label, idx in label_to_id.items()}