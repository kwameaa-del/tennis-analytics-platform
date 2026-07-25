from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tennis_analytics.exceptions import DataDownloadError

LOGGER = logging.getLogger(__name__)

URL_TEMPLATE = "https://www.tennis-data.co.uk/{year}{suffix}/{year}.xlsx"
VALID_TOURS = frozenset({"atp", "wta"})


def validate_tour(tour: str) -> str:
    """Normalize and validate a tennis tour name."""

    normalized = tour.strip().lower()

    if normalized not in VALID_TOURS:
        valid_tours = ", ".join(sorted(VALID_TOURS))
        raise ValueError(f"tour must be one of: {valid_tours}")

    return normalized


def _validate_download_options(
    years: Sequence[int],
    retries: int,
    retry_base_seconds: float,
) -> tuple[int, ...]:
    """Validate download options and return normalized years."""

    normalized_years = tuple(int(year) for year in years)

    if not normalized_years:
        raise ValueError("years must contain at least one year")

    if any(year < 1968 for year in normalized_years):
        raise ValueError("years must be 1968 or later")

    if len(normalized_years) != len(set(normalized_years)):
        raise ValueError("years must not contain duplicates")

    if retries < 1:
        raise ValueError("retries must be at least 1")

    if retry_base_seconds < 0:
        raise ValueError("retry_base_seconds must not be negative")

    return normalized_years


def _yearly_cache_path(output_dir: Path, tour: str, year: int) -> Path:
    """Return the cache path for one tour and season."""

    return output_dir / f"{tour}_raw_{year}.csv"


def _cache_is_usable(cache_path: Path) -> bool:
    """Return whether a cached CSV exists and contains readable rows."""

    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return False

    try:
        preview = pd.read_csv(cache_path, nrows=1)
    except (OSError, ValueError, pd.errors.ParserError):
        LOGGER.warning("Ignoring unreadable cache file: %s", cache_path)
        return False

    if preview.empty:
        LOGGER.warning("Ignoring empty cache file: %s", cache_path)
        return False

    return True


def _write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame without exposing a partially written destination."""

    temporary = destination.with_suffix(f"{destination.suffix}.tmp")

    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    except OSError as exc:
        raise DataDownloadError(
            f"Could not write CSV file {destination}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _download_year(
    tour: str,
    year: int,
    cache_path: Path,
    retries: int,
    retry_base_seconds: float,
) -> bool:
    """Download and cache one season, returning whether it succeeded."""

    suffix = "" if tour == "atp" else "w"
    url = URL_TEMPLATE.format(year=year, suffix=suffix)

    for attempt in range(1, retries + 1):
        try:
            LOGGER.info(
                "Downloading %s %s (attempt %s/%s)",
                tour.upper(),
                year,
                attempt,
                retries,
            )

            frame = pd.read_excel(url)

            if frame.empty:
                raise DataDownloadError(
                    f"Downloaded file for {tour.upper()} {year} was empty"
                )

            frame["Year"] = year
            _write_csv_atomically(frame, cache_path)

            LOGGER.info(
                "Cached %s rows for %s %s",
                len(frame),
                tour.upper(),
                year,
            )
            return True

        except (
            OSError,
            ValueError,
            pd.errors.ParserError,
            DataDownloadError,
        ) as exc:
            LOGGER.warning(
                "Download failed for %s %s on attempt %s/%s: %s",
                tour.upper(),
                year,
                attempt,
                retries,
                exc,
            )
        except Exception:
            # Spreadsheet and network backends may raise engine-specific
            # exceptions that do not share one stable public base class.
            LOGGER.exception(
                "Unexpected download failure for %s %s on attempt %s/%s",
                tour.upper(),
                year,
                attempt,
                retries,
            )

        if attempt < retries:
            delay_seconds = retry_base_seconds * attempt
            LOGGER.info(
                "Retrying %s %s in %.1f seconds",
                tour.upper(),
                year,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    return False


def _combine_cached_files(
    tour: str,
    years: Sequence[int],
    output_dir: Path,
) -> pd.DataFrame:
    """Load and combine all usable yearly cache files."""

    frames: list[pd.DataFrame] = []

    for year in years:
        cache_path = _yearly_cache_path(output_dir, tour, year)

        if not _cache_is_usable(cache_path):
            continue

        try:
            frame = pd.read_csv(cache_path, low_memory=False)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise DataDownloadError(
                f"Could not read cached {tour.upper()} file "
                f"for {year}: {exc}"
            ) from exc

        frames.append(frame)

    if not frames:
        raise DataDownloadError(
            f"No usable {tour.upper()} yearly files are available"
        )

    return pd.concat(frames, ignore_index=True)


def download_tour_data(
    tour: str,
    years: list[int] | tuple[int, ...],
    output_dir: Path,
    retries: int = 6,
    retry_base_seconds: float = 8.0,
) -> Path:
    """Download yearly tour data and create one combined raw CSV."""

    normalized_tour = validate_tour(tour)
    normalized_years = _validate_download_options(
        years,
        retries,
        retry_base_seconds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    failed_years: list[int] = []

    for year in normalized_years:
        cache_path = _yearly_cache_path(
            output_dir,
            normalized_tour,
            year,
        )

        if _cache_is_usable(cache_path):
            LOGGER.info(
                "Using cached %s %s data",
                normalized_tour.upper(),
                year,
            )
            continue

        succeeded = _download_year(
            tour=normalized_tour,
            year=year,
            cache_path=cache_path,
            retries=retries,
            retry_base_seconds=retry_base_seconds,
        )

        if not succeeded:
            failed_years.append(year)

    combined = _combine_cached_files(
        normalized_tour,
        normalized_years,
        output_dir,
    )

    output_path = output_dir / f"{normalized_tour}_raw.csv"
    _write_csv_atomically(combined, output_path)

    if failed_years:
        LOGGER.warning(
            "Pipeline continued without %s %s season(s): %s",
            len(failed_years),
            normalized_tour.upper(),
            failed_years,
        )

    LOGGER.info(
        "Saved %s combined %s rows to %s",
        len(combined),
        normalized_tour.upper(),
        output_path,
    )

    return output_path