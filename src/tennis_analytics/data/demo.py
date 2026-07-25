from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_analytics.data.download import validate_tour
from tennis_analytics.exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)

DEMO_PLAYERS = (
    "Alex Morgan",
    "Blake Carter",
    "Casey Jordan",
    "Drew Parker",
    "Emery Taylor",
    "Finley Reed",
    "Gray Murphy",
    "Hayden Brooks",
)

DEMO_SURFACES = (
    "Hard",
    "Clay",
    "Grass",
)


def _write_atomically(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write demo data without exposing a partially written file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    try:
        frame.to_csv(
            temporary_path,
            index=False,
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        raise DataValidationError(
            f"Could not write demo data to {output_path}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_demo_data(
    tour: str,
    output_path: Path,
    *,
    random_seed: int = 42,
    start_year: int = 2021,
    number_of_years: int = 5,
    matches_per_year: int = 80,
) -> Path:
    """Generate deterministic fictional tennis data for offline demos."""

    normalized_tour = validate_tour(tour)

    if number_of_years < 3:
        raise ValueError(
            "number_of_years must be at least 3"
        )

    if matches_per_year < 20:
        raise ValueError(
            "matches_per_year must be at least 20"
        )

    rng = np.random.default_rng(random_seed)

    player_strength = {
        player: strength
        for player, strength in zip(
            DEMO_PLAYERS,
            np.linspace(1.2, -1.2, len(DEMO_PLAYERS)),
            strict=True,
        )
    }

    records: list[dict[str, object]] = []

    for year in range(
        start_year,
        start_year + number_of_years,
    ):
        ranking_order = list(DEMO_PLAYERS)
        rng.shuffle(ranking_order)

        rankings = {
            player: rank
            for rank, player in enumerate(
                ranking_order,
                start=1,
            )
        }

        for match_number in range(matches_per_year):
            player_1, player_2 = rng.choice(
                DEMO_PLAYERS,
                size=2,
                replace=False,
            )

            strength_difference = (
                player_strength[str(player_1)]
                - player_strength[str(player_2)]
            )

            player_1_probability = 1.0 / (
                1.0 + np.exp(-strength_difference)
            )

            player_1_wins = (
                rng.random() < player_1_probability
            )

            winner = (
                str(player_1)
                if player_1_wins
                else str(player_2)
            )
            loser = (
                str(player_2)
                if player_1_wins
                else str(player_1)
            )

            winner_probability = (
                player_1_probability
                if player_1_wins
                else 1.0 - player_1_probability
            )

            winner_probability = float(
                np.clip(
                    winner_probability,
                    0.52,
                    0.88,
                )
            )

            margin = 1.06
            winner_odds = margin / winner_probability
            loser_odds = margin / (
                1.0 - winner_probability
            )

            match_date = (
                pd.Timestamp(year=year, month=1, day=1)
                + pd.Timedelta(days=match_number * 4)
            )

            records.append(
                {
                    "Date": match_date.date().isoformat(),
                    "Comment": "Completed",
                    "Winner": winner,
                    "Loser": loser,
                    "Surface": DEMO_SURFACES[
                        match_number % len(DEMO_SURFACES)
                    ],
                    "Best of": 3,
                    "WRank": rankings[winner],
                    "LRank": rankings[loser],
                    "Year": year,
                    "AvgW": round(winner_odds, 3),
                    "AvgL": round(loser_odds, 3),
                }
            )

    frame = pd.DataFrame.from_records(records)

    _write_atomically(
        frame,
        output_path,
    )

    LOGGER.info(
        "Generated %s fictional %s matches at %s",
        len(frame),
        normalized_tour.upper(),
        output_path,
    )

    return output_path