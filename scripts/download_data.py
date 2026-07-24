from __future__ import annotations

import argparse
import logging

from tennis_analytics.config import load_settings, project_root
from tennis_analytics.data.download import download_tour_data
from tennis_analytics.exceptions import TennisAnalyticsError
from tennis_analytics.logging_utils.setup import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and cache ATP/WTA match data")
    parser.add_argument("tour", choices=["atp", "wta"])
    args = parser.parse_args()
    try:
        settings = load_settings()
        root = project_root()
        configure_logging(level=settings.log_level, log_file=root / "logs" / "pipeline.log")
        path = download_tour_data(args.tour, settings.years, root / "data" / "raw")
        LOGGER.info("Data download complete: %s", path)
        return 0
    except TennisAnalyticsError as exc:
        LOGGER.error("Data download failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
