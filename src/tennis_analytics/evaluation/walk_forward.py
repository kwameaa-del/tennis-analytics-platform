from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analytics.exceptions import EvaluationError

LOGGER = logging.getLogger(__name__)

RANDOM_SEED = 42
MAX_ITERATIONS = 2_000

TENNIS_FEATURES = [
    "D_Elo",
    "D_SurfElo",
    "D_Form",
    "D_LogRank",
    "BestOf",
]

BENCHMARK_FEATURES = [
    *TENNIS_FEATURES,
    "BenchmarkProb_P1",
]

FEATURE_SETS = {
    "tennis_only": TENNIS_FEATURES,
    "tennis_plus_benchmark": BENCHMARK_FEATURES,
}


def _validate_regularization(c: float) -> float:
    """Validate and normalize logistic-regression regularization."""

    try:
        normalized = float(c)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must be a positive number") from exc

    if normalized <= 0:
        raise ValueError("c must be positive")

    return normalized


def _validate_probability_series(
    values: pd.Series,
    label: str,
) -> None:
    """Ensure probability values are strictly between zero and one."""

    if values.empty:
        raise EvaluationError(
            f"{label} probability series must not be empty"
        )

    if ((values <= 0) | (values >= 1)).any():
        raise EvaluationError(
            f"{label} probabilities must be strictly between 0 and 1"
        )


def _read_feature_data(features_path: Path) -> pd.DataFrame:
    """Read the feature dataset used for walk-forward evaluation."""

    if not features_path.is_file():
        raise EvaluationError(
            f"Features file not found: {features_path}"
        )

    try:
        frame = pd.read_csv(
            features_path,
            low_memory=False,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise EvaluationError(
            f"Could not read features file {features_path}: {exc}"
        ) from exc

    required_columns = [
        *BENCHMARK_FEATURES,
        "P1_Won",
        "Year",
        "Date",
    ]

    missing = set(required_columns) - set(frame.columns)

    if missing:
        raise EvaluationError(
            f"Missing evaluation columns: {sorted(missing)}"
        )

    return frame


def _prepare_evaluation_data(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Convert, validate, and order rows before evaluation."""

    prepared = frame.copy()

    prepared["Date"] = pd.to_datetime(
        prepared["Date"],
        errors="coerce",
    )

    numeric_columns = [
        *BENCHMARK_FEATURES,
        "P1_Won",
        "Year",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    required_columns = [
        *BENCHMARK_FEATURES,
        "P1_Won",
        "Year",
        "Date",
    ]

    prepared = prepared.dropna(
        subset=required_columns
    )

    prepared = prepared[
        prepared["P1_Won"].isin([0, 1])
    ]

    prepared = prepared.sort_values(
        ["Date", "Year"],
        kind="stable",
    ).reset_index(drop=True)

    if prepared.empty:
        raise EvaluationError(
            "No usable rows remained for walk-forward evaluation"
        )

    if not set(prepared["P1_Won"].unique()).issubset({0, 1}):
        raise EvaluationError(
            "P1_Won must contain only binary outcomes"
        )

    _validate_probability_series(
        prepared["BenchmarkProb_P1"],
        "Benchmark",
    )

    years = sorted(
        int(year)
        for year in prepared["Year"].unique()
    )

    if len(years) < 3:
        raise EvaluationError(
            "Walk-forward evaluation requires at least three years"
        )

    return prepared


def _create_pipeline(c: float) -> Pipeline:
    """Create a reproducible evaluation pipeline."""

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


def _calculate_metrics(
    outcomes: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float]:
    """Calculate probabilistic and classification metrics."""

    _validate_probability_series(
        probabilities,
        "Model",
    )

    predictions = probabilities >= 0.5

    return {
        "log_loss": float(
            log_loss(
                outcomes,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier": float(
            brier_score_loss(
                outcomes,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                outcomes,
                predictions,
            )
        ),
    }


def _benchmark_metrics(
    test_frame: pd.DataFrame,
) -> dict[str, float]:
    """Calculate metrics for the external benchmark probabilities."""

    probabilities = test_frame["BenchmarkProb_P1"]
    outcomes = test_frame["P1_Won"]

    return _calculate_metrics(
        outcomes,
        probabilities,
    )


def _evaluate_feature_set(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    c: float,
) -> dict[str, float]:
    """Train and evaluate one feature set."""

    pipeline = _create_pipeline(c)

    pipeline.fit(
        train_frame[list(feature_columns)],
        train_frame["P1_Won"],
    )

    probabilities = pd.Series(
        pipeline.predict_proba(
            test_frame[list(feature_columns)]
        )[:, 1],
        index=test_frame.index,
        dtype=float,
    )

    return _calculate_metrics(
        test_frame["P1_Won"],
        probabilities,
    )


def _build_evaluation_row(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_year: int,
    *,
    c: float,
) -> dict[str, float | int]:
    """Evaluate one walk-forward test year."""

    benchmark = _benchmark_metrics(
        test_frame
    )

    result: dict[str, float | int] = {
        "test_year": test_year,
        "n": len(test_frame),
        "benchmark_log_loss": benchmark["log_loss"],
        "benchmark_brier": benchmark["brier"],
        "benchmark_accuracy": benchmark["accuracy"],
    }

    for label, feature_columns in FEATURE_SETS.items():
        metrics = _evaluate_feature_set(
            train_frame,
            test_frame,
            feature_columns,
            c=c,
        )

        result[f"{label}_log_loss"] = metrics["log_loss"]
        result[f"{label}_brier"] = metrics["brier"]
        result[f"{label}_accuracy"] = metrics["accuracy"]

    return result


def evaluate_walk_forward(
    features_path: Path,
    c: float = 0.1,
) -> pd.DataFrame:
    """Train on prior years and evaluate each subsequent season."""

    regularization = _validate_regularization(c)
    raw_frame = _read_feature_data(features_path)
    frame = _prepare_evaluation_data(raw_frame)

    years = sorted(
        int(year)
        for year in frame["Year"].unique()
    )

    rows: list[dict[str, float | int]] = []

    for test_year in years[2:]:
        train_frame = frame[
            frame["Year"] < test_year
        ]
        test_frame = frame[
            frame["Year"] == test_year
        ]

        if train_frame.empty or test_frame.empty:
            LOGGER.warning(
                "Skipping %s because training or test data is empty",
                test_year,
            )
            continue

        if train_frame["P1_Won"].nunique() < 2:
            raise EvaluationError(
                f"Training data before {test_year} "
                "contains only one outcome class"
            )

        result = _build_evaluation_row(
            train_frame,
            test_frame,
            test_year,
            c=regularization,
        )

        rows.append(result)

        LOGGER.info(
            "Evaluated %s using %s training rows and %s test rows",
            test_year,
            len(train_frame),
            len(test_frame),
        )

    if not rows:
        raise EvaluationError(
            "No walk-forward windows could be evaluated"
        )

    return pd.DataFrame(rows)