from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from tennis_analytics.cli.commands import (
    run_build,
    run_download,
    run_evaluate,
    run_pipeline,
    run_train,
)
from tennis_analytics.config import load_settings, project_root
from tennis_analytics.logging_utils.setup import configure_logging

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="tennis",
        description=(
            "Run the Tennis Analytics Platform "
            "data and model workflows."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for command, description in (
        ("download", "Download historical match data"),
        ("build", "Build pre-match model features"),
        ("evaluate", "Run walk-forward evaluation"),
        ("train", "Train and persist the final model"),
    ):
        command_parser = subparsers.add_parser(
            command,
            help=description,
        )
        command_parser.add_argument(
            "tour",
            choices=("atp", "wta"),
            help="Tennis tour to process",
        )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the complete analytics pipeline",
    )
    pipeline_parser.add_argument(
        "tour",
        choices=("atp", "wta"),
        help="Tennis tour to process",
    )
    pipeline_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing raw data instead of downloading it",
    )

    return parser


def _configure_application_logging() -> None:
    """Configure logging from the project settings."""

    root = project_root()
    settings = load_settings()

    configure_logging(
        level=settings.log_level,
        log_file=root / "logs" / "pipeline.log",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _configure_application_logging()

        if args.command == "download":
            output = run_download(args.tour)
            print(f"Raw data saved to: {output}")

        elif args.command == "build":
            output = run_build(args.tour)
            print(f"Features saved to: {output}")

        elif args.command == "evaluate":
            output = run_evaluate(args.tour)
            print(f"Evaluation report saved to: {output}")

        elif args.command == "train":
            output = run_train(args.tour)
            print(f"Model saved to: {output}")

        elif args.command == "pipeline":
            outputs = run_pipeline(
                args.tour,
                skip_download=args.skip_download,
            )

            print("Pipeline completed successfully.")

            for name, path in outputs.items():
                print(f"{name}: {path}")

        else:
            parser.error(
                f"Unsupported command: {args.command}"
            )

    except Exception as exc:
        LOGGER.exception(
            "Command failed: %s",
            exc,
        )
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())