from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_benchmark_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark row must be a JSON object: {path}")
        rows.append(payload)
    return rows


def render_markdown(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    selected = columns or ["model", "dataset", "mAP", "micro_f1", "macro_f1", "checkpoint"]
    header = "| " + " | ".join(selected) + " |\n"
    separator = "|" + "|".join("---" for _ in selected) + "|\n"
    body = "".join(
        "| " + " | ".join(str(row.get(column, "")) for column in selected) + " |\n"
        for row in rows
    )
    return header + separator + body


def write_markdown(path: str | Path, rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows, columns=columns), encoding="utf-8")
