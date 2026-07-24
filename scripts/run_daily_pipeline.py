from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tennis_analytics.config import load_settings
from tennis_analytics.logging_utils.setup import configure_logging

LOGGER = logging.getLogger(__name__)


def run(command: list[str], root: Path) -> None:
    LOGGER.info("Running command: %s", " ".join(command))
    try:
        subprocess.run(command, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Command failed with exit code %s: %s", exc.returncode, " ".join(command))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh data, rebuild features, evaluate, and train models")
    parser.add_argument("--tours", nargs="+", choices=["atp", "wta"], default=["atp", "wta"])
    parser.add_argument("--skip-download", action="store_true", help="Use existing cached raw data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root / "config" / "settings.json")
    configure_logging(level=settings.log_level, log_file=root / "logs" / "pipeline.log")

    try:
        for tour in args.tours:
            if not args.skip_download:
                run([sys.executable, "scripts/download_data.py", tour], root)
            run([sys.executable, "scripts/build_features.py", tour], root)
            run([sys.executable, "scripts/evaluate.py", tour], root)
            run([sys.executable, "scripts/train_model.py", tour], root)
        LOGGER.info("Pipeline completed successfully for tours: %s", ", ".join(args.tours))
        return 0
    except subprocess.CalledProcessError:
        LOGGER.exception("Pipeline stopped after a failed stage")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
