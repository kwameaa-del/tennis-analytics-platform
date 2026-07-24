from __future__ import annotations

import argparse
import logging

from tennis_analytics.config import load_settings, project_root
from tennis_analytics.evaluation.walk_forward import evaluate_walk_forward
from tennis_analytics.exceptions import TennisAnalyticsError
from tennis_analytics.logging_utils.setup import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run year-by-year walk-forward evaluation")
    parser.add_argument("tour", choices=["atp", "wta"])
    args = parser.parse_args()
    try:
        root, settings = project_root(), load_settings()
        configure_logging(level=settings.log_level, log_file=root / "logs" / "pipeline.log")
        results = evaluate_walk_forward(
            root / "data" / "processed" / f"{args.tour}_features.csv",
            c=settings.logistic_regression_c,
        )
        output = root / "reports" / f"{args.tour}_walk_forward.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        LOGGER.info("Saved evaluation report to %s", output)
        print(results.to_string(index=False))
        return 0
    except TennisAnalyticsError as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
