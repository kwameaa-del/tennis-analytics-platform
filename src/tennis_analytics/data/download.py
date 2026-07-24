from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from tennis_analytics.exceptions import DataDownloadError

LOGGER = logging.getLogger(__name__)
URL_TEMPLATE = "https://www.tennis-data.co.uk/{year}{suffix}/{year}.xlsx"
VALID_TOURS = {"atp", "wta"}


def validate_tour(tour: str) -> str:
    normalized = tour.lower().strip()
    if normalized not in VALID_TOURS:
        raise ValueError("tour must be 'atp' or 'wta'")
    return normalized


def download_tour_data(
    tour: str,
    years: list[int] | tuple[int, ...],
    output_dir: Path,
    retries: int = 6,
    retry_base_seconds: float = 8.0,
) -> Path:
    """Download yearly ATP/WTA files, cache them, and build one master CSV."""
    tour = validate_tour(tour)
    if retries < 1:
        raise ValueError("retries must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if tour == "atp" else "w"
    failed: list[int] = []

    for year in years:
        cache = output_dir / f"{tour}_raw_{year}.csv"
        if cache.exists() and cache.stat().st_size > 0:
            LOGGER.info("Using cached %s %s data", tour.upper(), year)
            continue

        url = URL_TEMPLATE.format(year=year, suffix=suffix)
        for attempt in range(1, retries + 1):
            try:
                LOGGER.info("Downloading %s %s (attempt %s/%s)", tour.upper(), year, attempt, retries)
                frame = pd.read_excel(url)
                if frame.empty:
                    raise DataDownloadError(f"Downloaded file for {year} was empty")
                frame["Year"] = year
                temporary = cache.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(cache)
                LOGGER.info("Cached %s rows for %s %s", len(frame), tour.upper(), year)
                break
            except (OSError, ValueError, pd.errors.ParserError, DataDownloadError) as exc:
                LOGGER.warning("Download failed for %s %s: %s", tour.upper(), year, exc)
                if attempt == retries:
                    failed.append(year)
                else:
                    time.sleep(retry_base_seconds * attempt)
            except Exception as exc:  # library/network exceptions vary by engine
                LOGGER.exception("Unexpected download failure for %s %s", tour.upper(), year)
                if attempt == retries:
                    failed.append(year)
                else:
                    time.sleep(retry_base_seconds * attempt)

    available = [year for year in years if (output_dir / f"{tour}_raw_{year}.csv").exists()]
    if not available:
        raise DataDownloadError(f"No usable {tour.upper()} yearly files are available")

    try:
        combined = pd.concat(
            [pd.read_csv(output_dir / f"{tour}_raw_{year}.csv", low_memory=False) for year in available],
            ignore_index=True,
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise DataDownloadError(f"Could not combine cached {tour.upper()} files: {exc}") from exc

    output_path = output_dir / f"{tour}_raw.csv"
    combined.to_csv(output_path, index=False)

    if failed:
        LOGGER.warning("Pipeline continued with missing years: %s", failed)
    LOGGER.info("Saved %s combined rows to %s", len(combined), output_path)
    return output_path
