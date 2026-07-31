from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    """Injects the current request's correlation id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("voice_studio")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    file_handler = logging.FileHandler(log_dir / "voice_studio.log", encoding="utf-8")
    file_handler.addFilter(CorrelationIdFilter())
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.addFilter(CorrelationIdFilter())
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(correlation_id)s %(name)s: %(message)s")
    )
    root.addHandler(console_handler)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"voice_studio.{name}")


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    resolved = correlation_id or new_correlation_id()
    token = _correlation_id.set(resolved)
    try:
        yield resolved
    finally:
        _correlation_id.reset(token)
