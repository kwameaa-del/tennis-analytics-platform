from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure consistent console and optional rotating-file logging."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=5_000_000,
                backupCount=5,
                encoding="utf-8",
            )
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)
