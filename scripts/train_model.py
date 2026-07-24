from __future__ import annotations

import argparse
import logging

from tennis_analytics.config import load_settings, project_root
from tennis_analytics.logging_utils.setup import configure_logging
from tennis_analytics.models.train import train_tennis_model

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist the tennis-only forecasting model")
    parser.add_argument("tour", choices=["atp", "wta"])
    args = parser.parse_args()
    root, settings = project_root(), load_settings()
    configure_logging(level=settings.log_level, log_file=root / "logs" / "pipeline.log")
    artifact = train_tennis_model(
        root / "data" / "processed" / f"{args.tour}_features.csv",
        root / "models" / f"{args.tour}_model.joblib",
        c=settings.logistic_regression_c,
    )
    LOGGER.info("Trained %s model on %s rows through %s", args.tour.upper(), artifact.trained_rows, artifact.max_training_date)


if __name__ == "__main__":
    main()
