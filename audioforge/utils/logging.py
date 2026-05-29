from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)

LOG_EVENTS_TOTAL = Counter(
    "audioforge_log_events_total",
    "Total number of log events emitted by AudioForge.",
    ["level"],
)

TRAINING_STEP_DURATION_SECONDS = Histogram(
    "audioforge_training_step_duration_seconds",
    "Training step duration in seconds.",
)

ACTIVE_RUNS = Gauge(
    "audioforge_active_runs",
    "Number of active AudioForge runs.",
)


class JsonFormatter(logging.Formatter):
    """Small structured JSON formatter.

    Grafana is not a logger. Prometheus is not a logger either.
    This gives us JSON logs that can later be shipped to Loki/Grafana,
    while Prometheus metrics expose counters/histograms separately.
    Tiny distinction, massive reduction in future suffering.
    """

    def format(self, record: logging.LogRecord) -> str:
        LOG_EVENTS_TOTAL.labels(level=record.levelname).inc()

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


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    prometheus_enabled: bool = False,
    prometheus_host: str = "0.0.0.0",
    prometheus_port: int = 9090,
) -> None:
    """Configure app-wide logging.

    Args:
        level: Python logging level.
        json_logs: Use structured JSON logs if true.
        prometheus_enabled: Start Prometheus metrics server if true.
        prometheus_host: Metrics server host.
        prometheus_port: Metrics server port.
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

    if prometheus_enabled:
        start_http_server(port=prometheus_port, addr=prometheus_host)
        get_logger(__name__).info(
            "Prometheus metrics server started",
            extra={
                "extra_fields": {
                    "host": prometheus_host,
                    "port": prometheus_port,
                }
            },
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def set_run_id(run_id: str | None) -> None:
    _run_id.set(run_id)


class log_duration:
    """Context manager for timing blocks and optionally recording Prometheus histograms."""

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