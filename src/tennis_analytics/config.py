from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tennis_analytics.exceptions import ConfigurationError


@dataclass(frozen=True)
class Settings:
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
    return Path(__file__).resolve().parents[2]


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing configuration key: {key}")
    return mapping[key]


def load_settings(path: Path | None = None) -> Settings:
    settings_path = path or project_root() / "config" / "settings.json"
    if not settings_path.exists():
        raise ConfigurationError(f"Configuration file not found: {settings_path}")

    try:
        with settings_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {settings_path}: {exc}") from exc

    try:
        settings = Settings(
            years=tuple(int(y) for y in _require(raw, "years")),
            base_elo=float(_require(raw, "base_elo")),
            standard_k=float(_require(raw, "standard_k")),
            provisional_k=float(_require(raw, "provisional_k")),
            provisional_match_limit=int(_require(raw, "provisional_match_limit")),
            recent_form_window=int(_require(raw, "recent_form_window")),
            random_seed=int(_require(raw, "random_seed")),
            logistic_regression_c=float(_require(raw, "logistic_regression_c")),
            log_level=str(raw.get("log_level", "INFO")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration value: {exc}") from exc

    if len(settings.years) < 3:
        raise ConfigurationError("At least three years are required for walk-forward evaluation")
    if settings.recent_form_window < 1:
        raise ConfigurationError("recent_form_window must be positive")
    if settings.logistic_regression_c <= 0:
        raise ConfigurationError("logistic_regression_c must be positive")
    return settings
