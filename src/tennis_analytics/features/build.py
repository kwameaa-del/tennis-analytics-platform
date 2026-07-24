from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_analytics.exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)


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
    """Create pre-match, leak-resistant P1/P2 features from completed matches."""
    if not raw_path.exists():
        raise DataValidationError(f"Raw data file not found: {raw_path}")
    try:
        df = pd.read_csv(raw_path, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read {raw_path}: {exc}") from exc
    required = {"Date", "Comment", "Winner", "Loser", "Surface", "Best of", "WRank", "LRank", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    df = df.rename(columns={"Best of": "BestOf"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Comment"].eq("Completed")].copy()
    if df.empty:
        raise DataValidationError("No completed matches were available after filtering")

    psw = df["PSW"] if "PSW" in df else pd.Series(index=df.index, dtype=float)
    psl = df["PSL"] if "PSL" in df else pd.Series(index=df.index, dtype=float)
    avgw = df["AvgW"] if "AvgW" in df else pd.Series(index=df.index, dtype=float)
    avgl = df["AvgL"] if "AvgL" in df else pd.Series(index=df.index, dtype=float)
    df["Odds_W"] = psw.fillna(avgw)
    df["Odds_L"] = psl.fillna(avgl)
    df = df.dropna(subset=["Date", "Winner", "Loser", "Odds_W", "Odds_L"])

    elo: dict[str, float] = {}
    surface_elo: dict[tuple[str, str], float] = {}
    matches_played: dict[str, int] = defaultdict(int)
    recent: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=form_window))

    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        winner, loser, surface = row.Winner, row.Loser, row.Surface
        winner_elo, loser_elo = elo.get(winner, base_elo), elo.get(loser, base_elo)
        winner_surface = surface_elo.get((winner, surface), base_elo)
        loser_surface = surface_elo.get((loser, surface), base_elo)
        winner_form = float(np.mean(recent[winner])) if recent[winner] else np.nan
        loser_form = float(np.mean(recent[loser])) if recent[loser] else np.nan

        records.append({
            "Date": row.Date, "Surface": surface, "BestOf": row.BestOf,
            "Winner": winner, "Loser": loser, "WRank": row.WRank, "LRank": row.LRank,
            "W_Elo": winner_elo, "L_Elo": loser_elo,
            "W_SurfElo": winner_surface, "L_SurfElo": loser_surface,
            "W_Form10": winner_form, "L_Form10": loser_form,
            "Odds_W": row.Odds_W, "Odds_L": row.Odds_L, "Year": row.Year,
        })

        kw = provisional_k if matches_played[winner] < provisional_match_limit else standard_k
        kl = provisional_k if matches_played[loser] < provisional_match_limit else standard_k
        k = (kw + kl) / 2.0

        expected = 1.0 / (1.0 + 10 ** ((loser_elo - winner_elo) / 400.0))
        elo[winner] = winner_elo + k * (1.0 - expected)
        elo[loser] = loser_elo - k * (1.0 - expected)

        expected_surface = 1.0 / (1.0 + 10 ** ((loser_surface - winner_surface) / 400.0))
        surface_elo[(winner, surface)] = winner_surface + k * (1.0 - expected_surface)
        surface_elo[(loser, surface)] = loser_surface - k * (1.0 - expected_surface)

        matches_played[winner] += 1
        matches_played[loser] += 1
        recent[winner].append(1)
        recent[loser].append(0)

    prepared = pd.DataFrame(records)
    if prepared.empty:
        raise DataValidationError("No rows remained after required pre-match filtering")
    rng = np.random.default_rng(random_seed)
    flip = rng.random(len(prepared)) < 0.5

    out = pd.DataFrame({
        "Date": prepared["Date"], "Surface": prepared["Surface"], "BestOf": prepared["BestOf"],
        "P1": np.where(flip, prepared["Loser"], prepared["Winner"]),
        "P2": np.where(flip, prepared["Winner"], prepared["Loser"]),
        "P1_Rank": np.where(flip, prepared["LRank"], prepared["WRank"]),
        "P2_Rank": np.where(flip, prepared["WRank"], prepared["LRank"]),
        "P1_Elo": np.where(flip, prepared["L_Elo"], prepared["W_Elo"]),
        "P2_Elo": np.where(flip, prepared["W_Elo"], prepared["L_Elo"]),
        "P1_SurfElo": np.where(flip, prepared["L_SurfElo"], prepared["W_SurfElo"]),
        "P2_SurfElo": np.where(flip, prepared["W_SurfElo"], prepared["L_SurfElo"]),
        "P1_Form10": np.where(flip, prepared["L_Form10"], prepared["W_Form10"]),
        "P2_Form10": np.where(flip, prepared["W_Form10"], prepared["L_Form10"]),
        "P1_Odds": np.where(flip, prepared["Odds_L"], prepared["Odds_W"]),
        "P2_Odds": np.where(flip, prepared["Odds_W"], prepared["Odds_L"]),
        "P1_Won": np.where(flip, 0, 1), "Year": prepared["Year"],
    })

    inverse_1, inverse_2 = 1 / out["P1_Odds"], 1 / out["P2_Odds"]
    out["BenchmarkProb_P1"] = inverse_1 / (inverse_1 + inverse_2)
    out["D_Elo"] = out["P1_Elo"] - out["P2_Elo"]
    out["D_SurfElo"] = out["P1_SurfElo"] - out["P2_SurfElo"]
    out["D_Form"] = out["P1_Form10"] - out["P2_Form10"]
    out["D_LogRank"] = np.log(out["P2_Rank"].clip(lower=1)) - np.log(out["P1_Rank"].clip(lower=1))
    out = out.dropna(subset=["P1_Rank", "P2_Rank", "P1_Form10", "P2_Form10"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    LOGGER.info("Saved %s feature rows to %s; P1 win rate %.3f", len(out), output_path, out["P1_Won"].mean())
    return out
