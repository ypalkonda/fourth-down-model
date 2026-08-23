"""Pull play-by-play from nflverse and cache it locally as parquet.

nflreadpy hits the nflverse GitHub release assets. We cache per-season so a
rebuild does not re-download ~500k plays, and so results are reproducible
even if upstream data is revised.
"""

from __future__ import annotations

import polars as pl

from .config import RAW_DIR, SEASONS, ensure_dirs

# Columns we actually need. The raw feed is 372 columns wide; carrying all of
# it through every join wastes memory for no benefit.
KEEP_COLUMNS = [
    # identifiers / ordering
    "game_id", "play_id", "order_sequence", "season", "week", "season_type",
    "home_team", "away_team", "posteam", "defteam", "game_half", "qtr",
    "drive", "series",
    # pre-snap game state
    "down", "ydstogo", "yardline_100", "goal_to_go",
    "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_score", "defteam_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    # environment / market
    "roof", "surface", "temp", "wind", "spread_line", "total_line",
    # play description + outcome
    "play_type", "play_type_nfl", "desc", "yards_gained",
    "first_down", "third_down_converted",
    "fourth_down_converted", "fourth_down_failed",
    "rush_attempt", "pass_attempt", "qb_kneel", "qb_spike",
    "penalty", "aborted_play", "special_teams_play",
    # scoring events
    "sp", "touchdown", "td_team", "safety",
    "field_goal_attempt", "field_goal_result", "kick_distance",
    "extra_point_attempt", "extra_point_result",
    "two_point_attempt", "two_point_conv_result",
    "kickoff_attempt", "return_touchdown", "return_yards",
    # punts
    "punt_attempt", "punt_blocked", "punt_inside_twenty", "touchback",
    # turnovers
    "interception", "fumble_lost",
    # kickers (for the FG model's kicker embedding)
    "kicker_player_id", "kicker_player_name",
    # nflfastR's own estimates -- used ONLY as a benchmark, never as a feature
    "ep", "epa", "wp", "wpa", "vegas_wp",
    # final result, for win-probability labels
    "result", "home_score", "away_score",
    # drive-level context
    "fixed_drive_result", "series_result",
]


def _season_path(season: int):
    return RAW_DIR / f"pbp_{season}.parquet"


def download_season(season: int, force: bool = False) -> pl.DataFrame:
    """Fetch one season of pbp, caching to parquet."""
    import nflreadpy as nfl

    ensure_dirs()
    path = _season_path(season)
    if path.exists() and not force:
        return pl.read_parquet(path)

    df = nfl.load_pbp([season])
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"season {season} is missing expected columns: {missing}")

    df = df.select(KEEP_COLUMNS)
    df.write_parquet(path)
    return df


def load_pbp(seasons: list[int] | None = None, force: bool = False) -> pl.DataFrame:
    """Load and vertically concatenate the requested seasons."""
    seasons = seasons or SEASONS
    frames = [download_season(s, force=force) for s in seasons]
    # Schemas can drift slightly between seasons (e.g. a column that is all
    # null one year gets inferred as Null dtype); relaxed concat reconciles.
    return pl.concat(frames, how="vertical_relaxed")
