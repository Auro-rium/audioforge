from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import soundfile as sf


DCASE_SPLITS = Literal["train", "test", "eval", "unknown"]
DCASE_LABELS = Literal["normal", "anomaly", "unknown"]

DCASE_MANIFEST_COLUMNS = [
    "path",
    "split",
    "label",
    "machine_type",
    "section",
    "domain",
    "duration",
]


@dataclass(frozen=True)
class DCASEManifestRow:
    path: str
    split: str
    label: str
    machine_type: str
    section: str
    domain: str
    duration: float


@dataclass(frozen=True)
class DCASEManifestStats:
    rows: int
    train_rows: int
    test_rows: int
    eval_rows: int
    normal_rows: int
    anomaly_rows: int
    unknown_rows: int
    total_duration_hours: float
    machine_types: list[str]
    sections: list[str]
    domains: list[str]


def safe_audio_duration(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0:
            return 0.0
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def infer_label_from_name(path: str | Path) -> str:
    name = Path(path).name.lower()

    if "anomaly" in name or "abnormal" in name:
        return "anomaly"

    if "normal" in name:
        return "normal"

    return "unknown"


def infer_split_from_path(path: str | Path) -> str:
    parts = [part.lower() for part in Path(path).parts]

    if "train" in parts:
        return "train"

    if "test" in parts:
        return "test"

    if "eval" in parts or "evaluation" in parts:
        return "eval"

    return "unknown"


def infer_machine_type(path: str | Path, root: str | Path) -> str:
    audio_path = Path(path)
    root_path = Path(root)

    try:
        relative = audio_path.relative_to(root_path)
    except ValueError:
        relative = audio_path

    parts = list(relative.parts)

    for part in parts:
        lower = part.lower()
        if lower in {"train", "test", "eval", "evaluation", "source_train", "target_train"}:
            continue
        if part.endswith(".wav"):
            continue
        return part

    return "unknown"


def infer_section(path: str | Path) -> str:
    name = Path(path).name.lower()

    patterns = [
        r"section[_-]?(\d+)",
        r"section=([0-9]+)",
        r"id[_-]?(\d+)",
        r"id=([0-9]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return f"section_{int(match.group(1)):02d}"

    return "section_unknown"


def infer_domain(path: str | Path) -> str:
    name = Path(path).name.lower()
    parts = [part.lower() for part in Path(path).parts]

    if "source" in name or "source_train" in parts or "source_test" in parts:
        return "source"

    if "target" in name or "target_train" in parts or "target_test" in parts:
        return "target"

    return "unknown"


def discover_dcase_audio(root: str | Path) -> list[Path]:
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"DCASE root does not exist: {root_path}")

    audio_paths = sorted(root_path.rglob("*.wav"))

    if not audio_paths:
        raise FileNotFoundError(f"No .wav files found under DCASE root: {root_path}")

    return audio_paths


def build_dcase_manifest_rows(
    root: str | Path,
    *,
    absolute_paths: bool = False,
) -> list[DCASEManifestRow]:
    root_path = Path(root)
    audio_paths = discover_dcase_audio(root_path)

    rows: list[DCASEManifestRow] = []

    for audio_path in audio_paths:
        path_value: str

        if absolute_paths:
            path_value = str(audio_path.resolve())
        else:
            try:
                path_value = str(audio_path.relative_to(Path.cwd()))
            except ValueError:
                path_value = str(audio_path)

        rows.append(
            DCASEManifestRow(
                path=path_value,
                split=infer_split_from_path(audio_path),
                label=infer_label_from_name(audio_path),
                machine_type=infer_machine_type(audio_path, root_path),
                section=infer_section(audio_path),
                domain=infer_domain(audio_path),
                duration=safe_audio_duration(audio_path),
            )
        )

    return rows


def read_dcase_manifest(path: str | Path) -> list[DCASEManifestRow]:
    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"DCASE manifest not found: {manifest_path}")

    rows: list[DCASEManifestRow] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        missing = set(DCASE_MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"DCASE manifest missing columns: {sorted(missing)}")

        for raw in reader:
            rows.append(
                DCASEManifestRow(
                    path=raw["path"],
                    split=raw["split"],
                    label=raw["label"],
                    machine_type=raw["machine_type"],
                    section=raw["section"],
                    domain=raw["domain"],
                    duration=float(raw["duration"]),
                )
            )

    return rows


def write_dcase_manifest(path: str | Path, rows: Iterable[DCASEManifestRow]) -> DCASEManifestStats:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_list = list(rows)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DCASE_MANIFEST_COLUMNS)
        writer.writeheader()

        for row in row_list:
            payload = asdict(row)
            payload["duration"] = f"{row.duration:.6f}"
            writer.writerow(payload)

    return summarize_dcase_rows(row_list)


def summarize_dcase_rows(rows: list[DCASEManifestRow]) -> DCASEManifestStats:
    total_seconds = sum(row.duration for row in rows)

    return DCASEManifestStats(
        rows=len(rows),
        train_rows=sum(row.split == "train" for row in rows),
        test_rows=sum(row.split == "test" for row in rows),
        eval_rows=sum(row.split == "eval" for row in rows),
        normal_rows=sum(row.label == "normal" for row in rows),
        anomaly_rows=sum(row.label == "anomaly" for row in rows),
        unknown_rows=sum(row.label == "unknown" for row in rows),
        total_duration_hours=round(total_seconds / 3600.0, 4),
        machine_types=sorted({row.machine_type for row in rows}),
        sections=sorted({row.section for row in rows}),
        domains=sorted({row.domain for row in rows}),
    )


def build_dcase_manifest(
    root: str | Path,
    output_csv: str | Path,
    *,
    absolute_paths: bool = False,
) -> DCASEManifestStats:
    rows = build_dcase_manifest_rows(root, absolute_paths=absolute_paths)
    return write_dcase_manifest(output_csv, rows)


def split_dcase_rows(
    rows: list[DCASEManifestRow],
) -> tuple[list[DCASEManifestRow], list[DCASEManifestRow]]:
    train_rows = [row for row in rows if row.split == "train"]
    test_rows = [row for row in rows if row.split in {"test", "eval"}]

    return train_rows, test_rows


def label_to_int(label: str) -> int:
    if label == "normal":
        return 0

    if label == "anomaly":
        return 1

    raise ValueError(f"Cannot convert unknown label to int: {label}")
