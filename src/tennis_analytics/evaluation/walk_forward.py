from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from tennis_analytics.exceptions import EvaluationError

LOGGER = logging.getLogger(__name__)
TENNIS_FEATURES = ["D_Elo", "D_SurfElo", "D_Form", "D_LogRank", "BestOf"]
BENCHMARK_FEATURES = TENNIS_FEATURES + ["BenchmarkProb_P1"]


def _validate_probability_series(values: pd.Series, label: str) -> None:
    if ((values <= 0) | (values >= 1)).any():
        raise EvaluationError(f"{label} probabilities must be strictly between 0 and 1")


def evaluate_walk_forward(features_path: Path, c: float = 0.1) -> pd.DataFrame:
    """Train only on prior years and evaluate on each future year."""
    if c <= 0:
        raise ValueError("c must be positive")
    if not features_path.exists():
        raise EvaluationError(f"Features file not found: {features_path}")

    try:
        df = pd.read_csv(features_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise EvaluationError(f"Could not read features file: {exc}") from exc

    needed = BENCHMARK_FEATURES + ["P1_Won", "Year", "Date"]
    missing = set(needed) - set(df.columns)
    if missing:
        raise EvaluationError(f"Missing evaluation columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=needed).sort_values("Date")
    if not set(df["P1_Won"].unique()).issubset({0, 1}):
        raise EvaluationError("P1_Won must contain only binary outcomes")
    _validate_probability_series(df["BenchmarkProb_P1"], "Benchmark")

    years = sorted(df["Year"].unique())
    if len(years) < 3:
        raise EvaluationError("Walk-forward evaluation requires at least three years")

    rows: list[dict[str, float | int]] = []
    for test_year in years[2:]:
        train = df[df["Year"] < test_year]
        test = df[df["Year"] == test_year]
        if train.empty or test.empty:
            LOGGER.warning("Skipping %s because train or test data is empty", test_year)
            continue
        if train["P1_Won"].nunique() < 2:
            raise EvaluationError(f"Training data before {test_year} contains only one outcome class")

        row: dict[str, float | int] = {
            "test_year": int(test_year),
            "n": len(test),
            "benchmark_log_loss": log_loss(test["P1_Won"], test["BenchmarkProb_P1"]),
            "benchmark_brier": brier_score_loss(test["P1_Won"], test["BenchmarkProb_P1"]),
            "benchmark_accuracy": accuracy_score(test["P1_Won"], test["BenchmarkProb_P1"] >= 0.5),
        }
        feature_sets = {
            "tennis_only": TENNIS_FEATURES,
            "tennis_plus_benchmark": BENCHMARK_FEATURES,
        }
        for label, columns in feature_sets.items():
            scaler = StandardScaler()
            model = LogisticRegression(max_iter=2000, C=c, random_state=42)
            model.fit(scaler.fit_transform(train[columns]), train["P1_Won"])
            probabilities = model.predict_proba(scaler.transform(test[columns]))[:, 1]
            row[f"{label}_log_loss"] = log_loss(test["P1_Won"], probabilities)
            row[f"{label}_brier"] = brier_score_loss(test["P1_Won"], probabilities)
            row[f"{label}_accuracy"] = accuracy_score(test["P1_Won"], probabilities >= 0.5)
        LOGGER.info("Evaluated test year %s using %s matches", test_year, len(test))
        rows.append(row)

    if not rows:
        raise EvaluationError("No walk-forward windows could be evaluated")
    return pd.DataFrame(rows)
