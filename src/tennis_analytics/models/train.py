from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analytics.evaluation.walk_forward import TENNIS_FEATURES
from tennis_analytics.exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)

RANDOM_SEED = 42
MAX_ITERATIONS = 2_000


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Persisted model pipeline and its training metadata."""

    pipeline: Pipeline
    features: tuple[str, ...]
    trained_rows: int
    max_training_date: str


def _validate_regularization(c: float) -> float:
    """Validate and normalize logistic-regression regularization."""

    try:
        normalized = float(c)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must be a positive number") from exc

    if normalized <= 0:
        raise ValueError("c must be positive")

    return normalized


def _read_training_data(features_path: Path) -> pd.DataFrame:
    """Read and validate the feature dataset used for training."""

    if not features_path.is_file():
        raise DataValidationError(
            f"Features file not found: {features_path}"
        )

    try:
        frame = pd.read_csv(
            features_path,
            low_memory=False,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataValidationError(
            f"Could not read feature file {features_path}: {exc}"
        ) from exc

    required_columns = [
        *TENNIS_FEATURES,
        "P1_Won",
        "Date",
    ]

    missing = set(required_columns) - set(frame.columns)

    if missing:
        raise DataValidationError(
            f"Missing model-training columns: {sorted(missing)}"
        )

    return frame


def _prepare_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert, filter, and order rows before model fitting."""

    prepared = frame.copy()

    prepared["Date"] = pd.to_datetime(
        prepared["Date"],
        errors="coerce",
    )

    for feature in TENNIS_FEATURES:
        prepared[feature] = pd.to_numeric(
            prepared[feature],
            errors="coerce",
        )

    prepared["P1_Won"] = pd.to_numeric(
        prepared["P1_Won"],
        errors="coerce",
    )

    required_columns = [
        *TENNIS_FEATURES,
        "P1_Won",
        "Date",
    ]

    prepared = prepared.dropna(
        subset=required_columns
    )

    prepared = prepared[
        prepared["P1_Won"].isin([0, 1])
    ]

    prepared = prepared.sort_values(
        "Date",
        kind="stable",
    ).reset_index(drop=True)

    if prepared.empty:
        raise DataValidationError(
            "No usable rows remained for model training"
        )

    if prepared["P1_Won"].nunique() < 2:
        raise DataValidationError(
            "Training data must contain both outcome classes"
        )

    return prepared


def _create_pipeline(c: float) -> Pipeline:
    """Create the reproducible model-training pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=c,
                    max_iter=MAX_ITERATIONS,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _persist_artifact(
    artifact: ModelArtifact,
    output_path: Path,
) -> None:
    """Persist a model artifact without exposing a partial file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    try:
        joblib.dump(
            artifact,
            temporary_path,
        )
        temporary_path.replace(output_path)
    except (
        OSError,
        TypeError,
        ValueError,
        pickle.PickleError,
    ) as exc:
        raise DataValidationError(
            f"Could not persist model artifact "
            f"to {output_path}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def train_tennis_model(
    features_path: Path,
    output_path: Path,
    c: float = 0.1,
) -> ModelArtifact:
    """Train and persist a model using all valid historical rows."""

    regularization = _validate_regularization(c)
    raw_frame = _read_training_data(features_path)
    training_frame = _prepare_training_data(raw_frame)

    pipeline = _create_pipeline(
        regularization
    )

    pipeline.fit(
        training_frame[TENNIS_FEATURES],
        training_frame["P1_Won"],
    )

    max_training_date = (
        training_frame["Date"]
        .max()
        .date()
        .isoformat()
    )

    artifact = ModelArtifact(
        pipeline=pipeline,
        features=tuple(TENNIS_FEATURES),
        trained_rows=len(training_frame),
        max_training_date=max_training_date,
    )

    _persist_artifact(
        artifact,
        output_path,
    )

    LOGGER.info(
        "Persisted model trained on %s rows through %s to %s",
        artifact.trained_rows,
        artifact.max_training_date,
        output_path,
    )

    return artifact