from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

from tennis_analytics.exceptions import DataValidationError

FIELDNAMES: Final[tuple[str, ...]] = (
    "prediction_id",
    "timestamp_utc",
    "tour",
    "player_1",
    "player_2",
    "surface",
    "best_of",
    "model_p1_prob",
    "model_p2_prob",
    "benchmark_p1_prob",
    "benchmark_p2_prob",
    "status",
    "winner",
)

REQUIRED_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "tour",
    "player_1",
    "player_2",
    "surface",
    "best_of",
    "model_p1_prob",
    "benchmark_p1_prob",
)

VALID_TOURS: Final[frozenset[str]] = frozenset(
    {
        "atp",
        "wta",
    }
)

VALID_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "pending",
        "completed",
        "cancelled",
    }
)


def _require_fields(row: dict[str, object]) -> None:
    """Ensure every required prediction field is present."""

    missing = [
        field
        for field in REQUIRED_INPUT_FIELDS
        if field not in row
    ]

    if missing:
        raise DataValidationError(
            f"Missing prediction fields: {missing}"
        )


def _required_text(
    row: dict[str, object],
    field: str,
) -> str:
    """Return a normalized non-empty text value."""

    value = str(row[field]).strip()

    if not value:
        raise DataValidationError(
            f"{field} must not be empty"
        )

    return value


def _probability(
    row: dict[str, object],
    field: str,
) -> float:
    """Return a validated probability between zero and one."""

    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{field} must be numeric"
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise DataValidationError(
            f"{field} must be between 0 and 1"
        )

    return value


def _best_of(row: dict[str, object]) -> int:
    """Return a validated best-of match format."""

    try:
        value = int(row["best_of"])
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "best_of must be an integer"
        ) from exc

    if value not in {3, 5}:
        raise DataValidationError(
            "best_of must be either 3 or 5"
        )

    return value


def _timestamp_utc(row: dict[str, object]) -> str:
    """Return a normalized ISO-8601 UTC timestamp."""

    supplied = row.get("timestamp_utc")

    if supplied in (None, ""):
        return datetime.now(UTC).isoformat(
            timespec="seconds"
        )

    try:
        parsed = datetime.fromisoformat(
            str(supplied).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise DataValidationError(
            "timestamp_utc must be a valid ISO-8601 timestamp"
        ) from exc

    if parsed.tzinfo is None:
        raise DataValidationError(
            "timestamp_utc must include timezone information"
        )

    return parsed.astimezone(UTC).isoformat(
        timespec="seconds"
    )


def _prediction_id(row: dict[str, object]) -> str:
    """Return a supplied or generated prediction identifier."""

    supplied = str(
        row.get("prediction_id") or ""
    ).strip()

    if supplied:
        return supplied

    return f"pred-{uuid4().hex}"


def _validate_existing_header(path: Path) -> None:
    """Verify that an existing audit file uses the expected schema."""

    if not path.exists() or path.stat().st_size == 0:
        return

    try:
        with path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except OSError as exc:
        raise DataValidationError(
            f"Could not read prediction log {path}: {exc}"
        ) from exc

    if header != list(FIELDNAMES):
        raise DataValidationError(
            "Existing prediction log has an unexpected CSV schema"
        )


def _build_output_row(
    row: dict[str, object],
) -> dict[str, object]:
    """Validate input and create the persisted audit record."""

    _require_fields(row)

    tour = _required_text(
        row,
        "tour",
    ).lower()

    if tour not in VALID_TOURS:
        raise DataValidationError(
            "tour must be either 'atp' or 'wta'"
        )

    player_1 = _required_text(
        row,
        "player_1",
    )
    player_2 = _required_text(
        row,
        "player_2",
    )

    if player_1.casefold() == player_2.casefold():
        raise DataValidationError(
            "player_1 and player_2 must be different"
        )

    status = str(
        row.get("status", "pending")
    ).strip().lower()

    if status not in VALID_STATUSES:
        valid_statuses = ", ".join(
            sorted(VALID_STATUSES)
        )
        raise DataValidationError(
            f"status must be one of: {valid_statuses}"
        )

    winner = str(
        row.get("winner", "")
    ).strip()

    if status == "completed":
        valid_winners = {
            player_1.casefold(),
            player_2.casefold(),
        }

        if winner.casefold() not in valid_winners:
            raise DataValidationError(
                "A completed prediction must identify "
                "player_1 or player_2 as the winner"
            )

    model_p1 = _probability(
        row,
        "model_p1_prob",
    )
    benchmark_p1 = _probability(
        row,
        "benchmark_p1_prob",
    )

    return {
        "prediction_id": _prediction_id(row),
        "timestamp_utc": _timestamp_utc(row),
        "tour": tour,
        "player_1": player_1,
        "player_2": player_2,
        "surface": _required_text(
            row,
            "surface",
        ),
        "best_of": _best_of(row),
        "model_p1_prob": f"{model_p1:.6f}",
        "model_p2_prob": f"{1.0 - model_p1:.6f}",
        "benchmark_p1_prob": f"{benchmark_p1:.6f}",
        "benchmark_p2_prob": (
            f"{1.0 - benchmark_p1:.6f}"
        ),
        "status": status,
        "winner": winner,
    }


def append_prediction(
    path: Path,
    row: dict[str, object],
) -> str:
    """Validate and append one prediction audit record."""

    output = _build_output_row(row)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_existing_header(path)

    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    try:
        with path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
                extrasaction="raise",
            )

            if write_header:
                writer.writeheader()

            writer.writerow(output)
            handle.flush()

    except (OSError, csv.Error, ValueError) as exc:
        raise DataValidationError(
            f"Could not append prediction log {path}: {exc}"
        ) from exc

    return str(output["prediction_id"])