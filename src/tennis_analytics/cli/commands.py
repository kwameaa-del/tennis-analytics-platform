from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tennis_analytics.config import Settings, load_settings, project_root
from tennis_analytics.data.download import download_tour_data, validate_tour
from tennis_analytics.evaluation.walk_forward import evaluate_walk_forward
from tennis_analytics.features.build import build_features
from tennis_analytics.models.train import train_tennis_model

LOGGER = logging.getLogger(__name__)


def _paths(root: Path, tour: str) -> dict[str, Path]:
    """Return the standard project paths for one tour."""

    return {
        "raw_directory": root / "data" / "raw",
        "raw_file": root / "data" / "raw" / f"{tour}_raw.csv",
        "features_file": (
            root / "data" / "processed" / f"{tour}_features.csv"
        ),
        "evaluation_report": (
            root / "reports" / f"{tour}_walk_forward.csv"
        ),
        "model_file": (
            root / "models" / f"{tour}_model.joblib"
        ),
    }


def _context(tour: str) -> tuple[str, Path, Settings, dict[str, Path]]:
    """Load shared command configuration."""

    normalized_tour = validate_tour(tour)
    root = project_root()
    settings = load_settings()
    paths = _paths(root, normalized_tour)

    return normalized_tour, root, settings, paths


def _write_report_atomically(
    report: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write an evaluation report without exposing a partial file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    try:
        report.to_csv(
            temporary_path,
            index=False,
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_download(tour: str) -> Path:
    """Download and combine historical data for one tour."""

    normalized_tour, _, settings, paths = _context(tour)

    output_path = download_tour_data(
        normalized_tour,
        settings.years,
        paths["raw_directory"],
    )

    LOGGER.info(
        "Download completed: %s",
        output_path,
    )

    return output_path


def run_build(tour: str) -> Path:
    """Build model-ready features for one tour."""

    normalized_tour, _, settings, paths = _context(tour)

    features = build_features(
        paths["raw_file"],
        paths["features_file"],
        base_elo=settings.base_elo,
        standard_k=settings.standard_k,
        provisional_k=settings.provisional_k,
        provisional_match_limit=settings.provisional_match_limit,
        form_window=settings.recent_form_window,
        random_seed=settings.random_seed,
    )

    LOGGER.info(
        "Built %s %s feature rows",
        len(features),
        normalized_tour.upper(),
    )

    return paths["features_file"]


def run_evaluate(tour: str) -> Path:
    """Run walk-forward evaluation for one tour."""

    normalized_tour, _, settings, paths = _context(tour)

    report = evaluate_walk_forward(
        paths["features_file"],
        c=settings.logistic_regression_c,
    )

    _write_report_atomically(
        report,
        paths["evaluation_report"],
    )

    LOGGER.info(
        "Saved %s evaluation windows for %s to %s",
        len(report),
        normalized_tour.upper(),
        paths["evaluation_report"],
    )

    return paths["evaluation_report"]


def run_train(tour: str) -> Path:
    """Train and persist the final model for one tour."""

    normalized_tour, _, settings, paths = _context(tour)

    artifact = train_tennis_model(
        paths["features_file"],
        paths["model_file"],
        c=settings.logistic_regression_c,
    )

    LOGGER.info(
        "Trained %s model on %s rows through %s",
        normalized_tour.upper(),
        artifact.trained_rows,
        artifact.max_training_date,
    )

    return paths["model_file"]


def run_pipeline(
    tour: str,
    *,
    skip_download: bool = False,
) -> dict[str, Path]:
    """Run the complete data, evaluation, and training workflow."""

    normalized_tour = validate_tour(tour)

    LOGGER.info(
        "Starting full %s pipeline",
        normalized_tour.upper(),
    )

    outputs: dict[str, Path] = {}

    if not skip_download:
        outputs["raw_data"] = run_download(
            normalized_tour
        )

    outputs["features"] = run_build(
        normalized_tour
    )
    outputs["evaluation"] = run_evaluate(
        normalized_tour
    )
    outputs["model"] = run_train(
        normalized_tour
    )

    LOGGER.info(
        "Completed full %s pipeline",
        normalized_tour.upper(),
    )

    return outputs