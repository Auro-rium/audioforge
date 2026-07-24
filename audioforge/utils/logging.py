from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)


class JsonFormatter(logging.Formatter):
    """Small structured JSON formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        request_id = _request_id.get()
        run_id = _run_id.get()

        if request_id:
            payload["request_id"] = request_id

        if run_id:
            payload["run_id"] = run_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)

        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        prefix = f"[{record.levelname}] {record.name}: "
        return prefix + record.getMessage()


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure app-wide logging.

    Args:
        level: Python logging level.
        json_logs: Use structured JSON logs if true.
    """

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter

    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = PlainFormatter()

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def set_run_id(run_id: str | None) -> None:
    _run_id.set(run_id)


class log_duration:
    """Context manager for timing blocks and logging their duration."""

    def __init__(self, logger: logging.Logger, event: str, **fields: Any) -> None:
        self.logger = logger
        self.event = event
        self.fields = fields
        self.start_time: float | None = None

    def __enter__(self) -> log_duration:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.start_time is None:
            return

        duration = time.perf_counter() - self.start_time

        self.logger.info(
            self.event,
            extra={
                "extra_fields": {
                    **self.fields,
                    "duration_seconds": round(duration, 6),
                }
            },
        )
