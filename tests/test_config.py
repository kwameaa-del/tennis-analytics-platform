from __future__ import annotations

import json
from pathlib import Path

import pytest

from tennis_analytics.config import (
    PROJECT_ROOT_ENVIRONMENT_VARIABLE,
    load_settings,
    project_root,
)
from tennis_analytics.exceptions import ConfigurationError


def _valid_settings() -> dict[str, object]:
    """Return a complete valid settings dictionary."""

    return {
        "years": [2021, 2022, 2023],
        "base_elo": 1500,
        "standard_k": 32,
        "provisional_k": 48,
        "provisional_match_limit": 30,
        "recent_form_window": 10,
        "random_seed": 42,
        "logistic_regression_c": 0.1,
        "log_level": "INFO",
    }


def _create_project_root(root: Path) -> Path:
    """Create the files required for project-root detection."""

    config_directory = root / "config"
    config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (root / "pyproject.toml").write_text(
        "[project]\nname = 'test-project'\n",
        encoding="utf-8",
    )

    settings_path = config_directory / "settings.json"
    settings_path.write_text(
        json.dumps(_valid_settings()),
        encoding="utf-8",
    )

    return settings_path


def test_load_settings_validates_values(
    tmp_path: Path,
) -> None:
    """Settings validation should reject too few years."""

    path = tmp_path / "settings.json"

    settings = _valid_settings()
    settings["years"] = [2021, 2022]

    path.write_text(
        json.dumps(settings),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="At least three years",
    ):
        load_settings(path)


def test_load_settings_returns_valid_settings(
    tmp_path: Path,
) -> None:
    """A complete valid configuration should load successfully."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(_valid_settings()),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.years == (
        2021,
        2022,
        2023,
    )
    assert settings.base_elo == 1500
    assert settings.standard_k == 32
    assert settings.provisional_k == 48
    assert settings.provisional_match_limit == 30
    assert settings.recent_form_window == 10
    assert settings.random_seed == 42
    assert settings.logistic_regression_c == 0.1
    assert settings.log_level == "INFO"


def test_project_root_uses_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The explicit root environment variable should take priority."""

    project_directory = tmp_path / "configured-project"
    _create_project_root(project_directory)

    monkeypatch.setenv(
        PROJECT_ROOT_ENVIRONMENT_VARIABLE,
        str(project_directory),
    )

    assert project_root() == project_directory.resolve()


def test_project_root_rejects_invalid_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An invalid explicit project root should raise a clear error."""

    invalid_root = tmp_path / "missing-project-files"
    invalid_root.mkdir()

    monkeypatch.setenv(
        PROJECT_ROOT_ENVIRONMENT_VARIABLE,
        str(invalid_root),
    )

    with pytest.raises(
        ConfigurationError,
        match="does not point to a valid project root",
    ):
        project_root()


def test_project_root_detects_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project-root detection should work from the current directory."""

    project_directory = tmp_path / "project"
    nested_directory = project_directory / "src" / "package"

    nested_directory.mkdir(
        parents=True,
    )
    _create_project_root(project_directory)

    monkeypatch.delenv(
        PROJECT_ROOT_ENVIRONMENT_VARIABLE,
        raising=False,
    )
    monkeypatch.chdir(nested_directory)

    assert project_root() == project_directory.resolve()


def test_load_settings_uses_detected_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default settings loading should use the detected project root."""

    project_directory = tmp_path / "container-style-project"
    settings_path = _create_project_root(project_directory)

    monkeypatch.setenv(
        PROJECT_ROOT_ENVIRONMENT_VARIABLE,
        str(project_directory),
    )

    settings = load_settings()

    assert settings.years == (
        2021,
        2022,
        2023,
    )
    assert settings_path.exists()


def test_load_settings_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing settings file should raise ConfigurationError."""

    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        ConfigurationError,
        match="Configuration file not found",
    ):
        load_settings(missing_path)


def test_load_settings_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should raise ConfigurationError."""

    path = tmp_path / "settings.json"
    path.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Invalid JSON",
    ):
        load_settings(path)


def test_load_settings_rejects_non_object_root(
    tmp_path: Path,
) -> None:
    """The configuration root must be a JSON object."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            [
                "not",
                "an",
                "object",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Configuration root must be a JSON object",
    ):
        load_settings(path)


def test_load_settings_rejects_missing_required_key(
    tmp_path: Path,
) -> None:
    """Missing required settings should raise a clear error."""

    path = tmp_path / "settings.json"
    settings = _valid_settings()
    del settings["base_elo"]

    path.write_text(
        json.dumps(settings),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Missing configuration key: base_elo",
    ):
        load_settings(path)