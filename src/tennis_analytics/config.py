from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tennis_analytics.exceptions import ConfigurationError

VALID_LOG_LEVELS = frozenset(logging.getLevelNamesMapping())


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for the tennis analytics pipeline."""

    years: tuple[int, ...]
    base_elo: float
    standard_k: float
    provisional_k: float
    provisional_match_limit: int
    recent_form_window: int
    random_seed: int
    logistic_regression_c: float
    log_level: str = "INFO"


def project_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    """Return a required configuration value or raise a clear error."""

    if key not in mapping:
        raise ConfigurationError(f"Missing configuration key: {key}")

    return mapping[key]


def _load_json(settings_path: Path) -> Mapping[str, Any]:
    """Read and validate the top-level JSON configuration object."""

    try:
        with settings_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Configuration file not found: {settings_path}"
        ) from exc
    except PermissionError as exc:
        raise ConfigurationError(
            f"Permission denied while reading configuration: {settings_path}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configuration file {settings_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in {settings_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            "Configuration root must be a JSON object"
        )

    return raw


def _validate_settings(settings: Settings) -> None:
    """Validate relationships and allowed ranges between settings."""

    if len(settings.years) < 3:
        raise ConfigurationError(
            "At least three years are required for walk-forward evaluation"
        )

    if len(settings.years) != len(set(settings.years)):
        raise ConfigurationError("Configuration years must not contain duplicates")

    if tuple(sorted(settings.years)) != settings.years:
        raise ConfigurationError(
            "Configuration years must be listed in ascending order"
        )

    if settings.base_elo <= 0:
        raise ConfigurationError("base_elo must be positive")

    if settings.standard_k <= 0:
        raise ConfigurationError("standard_k must be positive")

    if settings.provisional_k <= 0:
        raise ConfigurationError("provisional_k must be positive")

    if settings.provisional_match_limit < 1:
        raise ConfigurationError(
            "provisional_match_limit must be positive"
        )

    if settings.recent_form_window < 1:
        raise ConfigurationError("recent_form_window must be positive")

    if settings.random_seed < 0:
        raise ConfigurationError("random_seed must not be negative")

    if settings.logistic_regression_c <= 0:
        raise ConfigurationError(
            "logistic_regression_c must be positive"
        )

    if settings.log_level not in VALID_LOG_LEVELS:
        valid_levels = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigurationError(
            f"Invalid log_level '{settings.log_level}'. "
            f"Expected one of: {valid_levels}"
        )


def load_settings(path: Path | None = None) -> Settings:
    """Load, convert, and validate application settings from JSON."""

    settings_path = path or project_root() / "config" / "settings.json"
    raw = _load_json(settings_path)

    try:
        years_value = _require(raw, "years")

        if isinstance(years_value, (str, bytes)) or not isinstance(
            years_value, list
        ):
            raise ConfigurationError("years must be a JSON array")

        settings = Settings(
            years=tuple(int(year) for year in years_value),
            base_elo=float(_require(raw, "base_elo")),
            standard_k=float(_require(raw, "standard_k")),
            provisional_k=float(_require(raw, "provisional_k")),
            provisional_match_limit=int(
                _require(raw, "provisional_match_limit")
            ),
            recent_form_window=int(_require(raw, "recent_form_window")),
            random_seed=int(_require(raw, "random_seed")),
            logistic_regression_c=float(
                _require(raw, "logistic_regression_c")
            ),
            log_level=str(raw.get("log_level", "INFO")).upper(),
        )
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Invalid configuration value: {exc}"
        ) from exc

    _validate_settings(settings)
    return settings