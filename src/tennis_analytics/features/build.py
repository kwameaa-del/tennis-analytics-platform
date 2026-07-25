from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_analytics.exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Date",
    "Comment",
    "Winner",
    "Loser",
    "Surface",
    "Best of",
    "WRank",
    "LRank",
    "Year",
}


def _validate_options(
    *,
    base_elo: float,
    standard_k: float,
    provisional_k: float,
    provisional_match_limit: int,
    form_window: int,
    random_seed: int,
) -> None:
    """Validate feature-engineering configuration values."""

    if base_elo <= 0:
        raise ValueError("base_elo must be positive")

    if standard_k <= 0:
        raise ValueError("standard_k must be positive")

    if provisional_k <= 0:
        raise ValueError("provisional_k must be positive")

    if provisional_match_limit < 1:
        raise ValueError("provisional_match_limit must be positive")

    if form_window < 1:
        raise ValueError("form_window must be positive")

    if random_seed < 0:
        raise ValueError("random_seed must not be negative")


def _read_raw_matches(raw_path: Path) -> pd.DataFrame:
    """Read the raw match file and verify its required schema."""

    if not raw_path.is_file():
        raise DataValidationError(f"Raw data file not found: {raw_path}")

    try:
        frame = pd.read_csv(raw_path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataValidationError(
            f"Could not read {raw_path}: {exc}"
        ) from exc

    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not ({"PSW", "AvgW"} & set(frame.columns)):
        raise DataValidationError(
            "At least one winner-odds column is required: PSW or AvgW"
        )

    if not ({"PSL", "AvgL"} & set(frame.columns)):
        raise DataValidationError(
            "At least one loser-odds column is required: PSL or AvgL"
        )

    return frame


def _odds_series(
    frame: pd.DataFrame,
    primary: str,
    fallback: str,
) -> pd.Series:
    """Return numeric odds using a primary and fallback column."""

    primary_values = (
        pd.to_numeric(frame[primary], errors="coerce")
        if primary in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )

    fallback_values = (
        pd.to_numeric(frame[fallback], errors="coerce")
        if fallback in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )

    return primary_values.fillna(fallback_values)


def _prepare_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw rows and retain usable completed matches."""

    prepared = frame.copy()

    # Preserve the original order as a deterministic tie-breaker when
    # multiple matches have the same date.
    prepared["_source_order"] = np.arange(len(prepared))
    prepared = prepared.rename(columns={"Best of": "BestOf"})

    prepared["Date"] = pd.to_datetime(
        prepared["Date"],
        errors="coerce",
    )
    prepared["WRank"] = pd.to_numeric(
        prepared["WRank"],
        errors="coerce",
    )
    prepared["LRank"] = pd.to_numeric(
        prepared["LRank"],
        errors="coerce",
    )
    prepared["BestOf"] = pd.to_numeric(
        prepared["BestOf"],
        errors="coerce",
    )
    prepared["Year"] = pd.to_numeric(
        prepared["Year"],
        errors="coerce",
    )

    prepared["Odds_W"] = _odds_series(
        prepared,
        "PSW",
        "AvgW",
    )
    prepared["Odds_L"] = _odds_series(
        prepared,
        "PSL",
        "AvgL",
    )

    completed = (
        prepared["Comment"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("completed")
    )

    prepared = prepared.loc[completed].copy()

    for column in ("Winner", "Loser", "Surface"):
        prepared[column] = (
            prepared[column]
            .astype("string")
            .str.strip()
        )
        prepared.loc[
            prepared[column].eq(""),
            column,
        ] = pd.NA

    prepared = prepared.dropna(
        subset=[
            "Date",
            "Winner",
            "Loser",
            "Surface",
            "BestOf",
            "WRank",
            "LRank",
            "Year",
            "Odds_W",
            "Odds_L",
        ]
    )

    prepared = prepared[
        (prepared["WRank"] >= 1)
        & (prepared["LRank"] >= 1)
        & (prepared["Odds_W"] > 0)
        & (prepared["Odds_L"] > 0)
        & prepared["Winner"].ne(prepared["Loser"])
    ]

    prepared = prepared.sort_values(
        ["Date", "_source_order"],
        kind="stable",
    ).reset_index(drop=True)

    if prepared.empty:
        raise DataValidationError(
            "No usable completed matches remained after validation"
        )

    return prepared


def _elo_expected(
    player_rating: float,
    opponent_rating: float,
) -> float:
    """Return the expected win probability from two Elo ratings."""

    rating_difference = opponent_rating - player_rating

    return 1.0 / (
        1.0 + 10.0 ** (rating_difference / 400.0)
    )


def _match_k_factor(
    winner: str,
    loser: str,
    matches_played: dict[str, int],
    *,
    standard_k: float,
    provisional_k: float,
    provisional_match_limit: int,
) -> float:
    """Return the average K-factor for the two players."""

    winner_k = (
        provisional_k
        if matches_played[winner] < provisional_match_limit
        else standard_k
    )

    loser_k = (
        provisional_k
        if matches_played[loser] < provisional_match_limit
        else standard_k
    )

    return (winner_k + loser_k) / 2.0


def _build_pre_match_records(
    matches: pd.DataFrame,
    *,
    base_elo: float,
    standard_k: float,
    provisional_k: float,
    provisional_match_limit: int,
    form_window: int,
) -> pd.DataFrame:
    """Create records using only state known before each match."""

    elo: dict[str, float] = {}
    surface_elo: dict[tuple[str, str], float] = {}
    matches_played: dict[str, int] = defaultdict(int)

    recent: dict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=form_window)
    )

    records: list[dict[str, Any]] = []

    for row in matches.itertuples(index=False):
        winner = str(row.Winner)
        loser = str(row.Loser)
        surface = str(row.Surface)

        winner_elo = elo.get(winner, base_elo)
        loser_elo = elo.get(loser, base_elo)

        winner_surface_elo = surface_elo.get(
            (winner, surface),
            base_elo,
        )
        loser_surface_elo = surface_elo.get(
            (loser, surface),
            base_elo,
        )

        winner_form = (
            float(np.mean(recent[winner]))
            if recent[winner]
            else np.nan
        )
        loser_form = (
            float(np.mean(recent[loser]))
            if recent[loser]
            else np.nan
        )

        # This record is created before the current match updates
        # any Elo rating, match count, or recent-form history.
        records.append(
            {
                "Date": row.Date,
                "Surface": surface,
                "BestOf": row.BestOf,
                "Winner": winner,
                "Loser": loser,
                "WRank": row.WRank,
                "LRank": row.LRank,
                "W_Elo": winner_elo,
                "L_Elo": loser_elo,
                "W_SurfElo": winner_surface_elo,
                "L_SurfElo": loser_surface_elo,
                "W_Form": winner_form,
                "L_Form": loser_form,
                "Odds_W": row.Odds_W,
                "Odds_L": row.Odds_L,
                "Year": int(row.Year),
            }
        )

        k_factor = _match_k_factor(
            winner,
            loser,
            matches_played,
            standard_k=standard_k,
            provisional_k=provisional_k,
            provisional_match_limit=provisional_match_limit,
        )

        expected = _elo_expected(
            winner_elo,
            loser_elo,
        )
        rating_change = k_factor * (1.0 - expected)

        elo[winner] = winner_elo + rating_change
        elo[loser] = loser_elo - rating_change

        expected_surface = _elo_expected(
            winner_surface_elo,
            loser_surface_elo,
        )
        surface_change = (
            k_factor * (1.0 - expected_surface)
        )

        surface_elo[
            (winner, surface)
        ] = winner_surface_elo + surface_change

        surface_elo[
            (loser, surface)
        ] = loser_surface_elo - surface_change

        matches_played[winner] += 1
        matches_played[loser] += 1

        recent[winner].append(1)
        recent[loser].append(0)

    return pd.DataFrame.from_records(records)


def _frame_players(
    prepared: pd.DataFrame,
    *,
    random_seed: int,
) -> pd.DataFrame:
    """Assign winner and loser to reproducible P1/P2 positions."""

    rng = np.random.default_rng(random_seed)
    loser_is_p1 = rng.random(len(prepared)) < 0.5

    framed = pd.DataFrame(
        {
            "Date": prepared["Date"],
            "Surface": prepared["Surface"],
            "BestOf": prepared["BestOf"],
            "P1": np.where(
                loser_is_p1,
                prepared["Loser"],
                prepared["Winner"],
            ),
            "P2": np.where(
                loser_is_p1,
                prepared["Winner"],
                prepared["Loser"],
            ),
            "P1_Rank": np.where(
                loser_is_p1,
                prepared["LRank"],
                prepared["WRank"],
            ),
            "P2_Rank": np.where(
                loser_is_p1,
                prepared["WRank"],
                prepared["LRank"],
            ),
            "P1_Elo": np.where(
                loser_is_p1,
                prepared["L_Elo"],
                prepared["W_Elo"],
            ),
            "P2_Elo": np.where(
                loser_is_p1,
                prepared["W_Elo"],
                prepared["L_Elo"],
            ),
            "P1_SurfElo": np.where(
                loser_is_p1,
                prepared["L_SurfElo"],
                prepared["W_SurfElo"],
            ),
            "P2_SurfElo": np.where(
                loser_is_p1,
                prepared["W_SurfElo"],
                prepared["L_SurfElo"],
            ),
            # These names are retained for compatibility with the
            # existing output schema, even when form_window is changed.
            "P1_Form10": np.where(
                loser_is_p1,
                prepared["L_Form"],
                prepared["W_Form"],
            ),
            "P2_Form10": np.where(
                loser_is_p1,
                prepared["W_Form"],
                prepared["L_Form"],
            ),
            "P1_Odds": np.where(
                loser_is_p1,
                prepared["Odds_L"],
                prepared["Odds_W"],
            ),
            "P2_Odds": np.where(
                loser_is_p1,
                prepared["Odds_W"],
                prepared["Odds_L"],
            ),
            "P1_Won": np.where(
                loser_is_p1,
                0,
                1,
            ),
            "Year": prepared["Year"],
        }
    )

    inverse_p1 = 1.0 / framed["P1_Odds"]
    inverse_p2 = 1.0 / framed["P2_Odds"]

    framed["BenchmarkProb_P1"] = (
        inverse_p1 / (inverse_p1 + inverse_p2)
    )

    framed["D_Elo"] = (
        framed["P1_Elo"] - framed["P2_Elo"]
    )
    framed["D_SurfElo"] = (
        framed["P1_SurfElo"]
        - framed["P2_SurfElo"]
    )
    framed["D_Form"] = (
        framed["P1_Form10"]
        - framed["P2_Form10"]
    )
    framed["D_LogRank"] = (
        np.log(framed["P2_Rank"])
        - np.log(framed["P1_Rank"])
    )

    return framed.dropna(
        subset=[
            "P1_Rank",
            "P2_Rank",
            "P1_Form10",
            "P2_Form10",
        ]
    ).reset_index(drop=True)


def _write_features_atomically(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write feature data without exposing a partial file."""

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
            f"Could not write feature file "
            f"{output_path}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def build_features(
    raw_path: Path,
    output_path: Path,
    *,
    base_elo: float = 1500.0,
    standard_k: float = 32.0,
    provisional_k: float = 48.0,
    provisional_match_limit: int = 30,
    form_window: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create reproducible pre-match P1/P2 features."""

    _validate_options(
        base_elo=base_elo,
        standard_k=standard_k,
        provisional_k=provisional_k,
        provisional_match_limit=provisional_match_limit,
        form_window=form_window,
        random_seed=random_seed,
    )

    raw_matches = _read_raw_matches(raw_path)
    matches = _prepare_matches(raw_matches)

    prepared = _build_pre_match_records(
        matches,
        base_elo=base_elo,
        standard_k=standard_k,
        provisional_k=provisional_k,
        provisional_match_limit=provisional_match_limit,
        form_window=form_window,
    )

    features = _frame_players(
        prepared,
        random_seed=random_seed,
    )

    if features.empty:
        raise DataValidationError(
            "No feature rows remained after pre-match filtering"
        )

    _write_features_atomically(
        features,
        output_path,
    )

    LOGGER.info(
        "Saved %s feature rows to %s; P1 win rate %.3f",
        len(features),
        output_path,
        features["P1_Won"].mean(),
    )

    return features