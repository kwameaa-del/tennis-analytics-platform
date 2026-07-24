from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analytics.evaluation.walk_forward import TENNIS_FEATURES
from tennis_analytics.exceptions import DataValidationError


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Pipeline
    features: tuple[str, ...]
    trained_rows: int
    max_training_date: str


def train_tennis_model(features_path: Path, output_path: Path, c: float = 0.1) -> ModelArtifact:
    """Train the tennis-only model on all available historical rows."""
    if not features_path.exists():
        raise DataValidationError(f"Features file not found: {features_path}")
    frame = pd.read_csv(features_path)
    required = TENNIS_FEATURES + ["P1_Won", "Date"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise DataValidationError(f"Missing model-training columns: {sorted(missing)}")
    frame = frame.dropna(subset=required)
    if frame.empty or frame["P1_Won"].nunique() < 2:
        raise DataValidationError("Training data must contain both outcome classes")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=c, random_state=42)),
    ])
    pipeline.fit(frame[TENNIS_FEATURES], frame["P1_Won"])
    artifact = ModelArtifact(
        pipeline=pipeline,
        features=tuple(TENNIS_FEATURES),
        trained_rows=len(frame),
        max_training_date=str(pd.to_datetime(frame["Date"]).max().date()),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact
