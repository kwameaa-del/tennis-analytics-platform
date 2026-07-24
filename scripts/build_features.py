from __future__ import annotations

import argparse
import logging

from tennis_analytics.config import load_settings, project_root
from tennis_analytics.exceptions import TennisAnalyticsError
from tennis_analytics.features.build import build_features
from tennis_analytics.logging_utils.setup import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leak-resistant pre-match features")
    parser.add_argument("tour", choices=["atp", "wta"])
    args = parser.parse_args()
    try:
        root, settings = project_root(), load_settings()
        configure_logging(level=settings.log_level, log_file=root / "logs" / "pipeline.log")
        frame = build_features(
            root / "data" / "raw" / f"{args.tour}_raw.csv",
            root / "data" / "processed" / f"{args.tour}_features.csv",
            base_elo=settings.base_elo,
            standard_k=settings.standard_k,
            provisional_k=settings.provisional_k,
            provisional_match_limit=settings.provisional_match_limit,
            form_window=settings.recent_form_window,
            random_seed=settings.random_seed,
        )
        LOGGER.info("Feature build complete: %s rows; P1 win rate %.3f", len(frame), frame["P1_Won"].mean())
        return 0
    except TennisAnalyticsError as exc:
        LOGGER.error("Feature build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
