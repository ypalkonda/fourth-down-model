"""Target construction for the expected-points and win-probability models.

Two labels are built here.

`next_score_type` -- the EP target. For every play, the next scoring event in
the same half, expressed from the possession team's point of view. This is the
Carter/Machol "next score" formulation that every modern EP model descends
from: the value of a game state is the points you expect to net from it before
the half ends.

`posteam_won` -- the WP target. Did the team with the ball go on to win?

Both are constructed from raw event columns rather than borrowed from
nflfastR's precomputed `ep`/`wp`, which are kept aside strictly as a
benchmark.
"""

from __future__ import annotations

import polars as pl

from .config import (
    FIRST_17_GAME_SEASON,
    FIRST_DYNAMIC_KICKOFF_SEASON,
    FIRST_NEW_OT_SEASON,
)

# --------------------------------------------------------------------------
# Scoring events
# --------------------------------------------------------------------------
# A "scoring event" for EP purposes is a touchdown, a made field goal, or a
# safety. PATs and two-point tries are deliberately NOT events: the TD class
# is valued at 7, which already folds in the average PAT outcome. Counting
# them separately would double-count the point-after.


def _tag_scoring_events(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the points and scoring team for plays that produced a score."""
    is_pat = (pl.col("extra_point_attempt") == 1) | (pl.col("two_point_attempt") == 1)

    pts = (
        pl.when(is_pat)
        .then(None)
        .when(pl.col("touchdown") == 1)
        .then(pl.lit(7.0))
        .when(pl.col("field_goal_result") == "made")
        .then(pl.lit(3.0))
        .when(pl.col("safety") == 1)
        .then(pl.lit(2.0))
        .otherwise(None)
        .alias("_score_pts")
    )

    # Who gets the points:
    #   touchdown -> `td_team` (handles pick-sixes / punt returns, where the
    #                defense scores and td_team != posteam)
    #   made FG   -> the kicking team, i.e. posteam
    #   safety    -> the DEFENSE. On a safety the offense is tackled in its
    #                own end zone, so the two points go to defteam.
    team = (
        pl.when(is_pat)
        .then(None)
        .when(pl.col("touchdown") == 1)
        .then(pl.col("td_team"))
        .when(pl.col("field_goal_result") == "made")
        .then(pl.col("posteam"))
        .when(pl.col("safety") == 1)
        .then(pl.col("defteam"))
        .otherwise(None)
        .alias("_score_team")
    )

    return df.with_columns([pts, team])


def add_next_score_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Backward-fill the next scoring event within each half.

    The fill is *inclusive* of the current row: a play that is itself a
    touchdown has "Touchdown" as its next score, which is correct, because EP
    is the value of the state before the snap and that snap produced the score.
    """
    df = df.sort(["game_id", "order_sequence", "play_id"])
    df = _tag_scoring_events(df)

    df = df.with_columns(
        [
            pl.col("_score_pts")
            .fill_null(strategy="backward")
            .over(["game_id", "game_half"])
            .alias("_next_pts"),
            pl.col("_score_team")
            .fill_null(strategy="backward")
            .over(["game_id", "game_half"])
            .alias("_next_team"),
            # Drive index of the scoring play, for the sample weights below.
            pl.when(pl.col("_score_pts").is_not_null())
            .then(pl.col("drive"))
            .otherwise(None)
            .fill_null(strategy="backward")
            .over(["game_id", "game_half"])
            .alias("_score_drive"),
        ]
    )

    scored_for = pl.col("_next_team") == pl.col("posteam")

    df = df.with_columns(
        pl.when(pl.col("_next_team").is_null())
        .then(pl.lit("No_Score"))
        .when(scored_for & (pl.col("_next_pts") == 7.0))
        .then(pl.lit("Touchdown"))
        .when(scored_for & (pl.col("_next_pts") == 3.0))
        .then(pl.lit("Field_Goal"))
        .when(scored_for & (pl.col("_next_pts") == 2.0))
        .then(pl.lit("Safety"))
        .when(pl.col("_next_pts") == 7.0)
        .then(pl.lit("Opp_Touchdown"))
        .when(pl.col("_next_pts") == 3.0)
        .then(pl.lit("Opp_Field_Goal"))
        .when(pl.col("_next_pts") == 2.0)
        .then(pl.lit("Opp_Safety"))
        .otherwise(pl.lit("No_Score"))
        .alias("next_score_type")
    )

    return df.drop(["_score_pts", "_score_team", "_next_pts", "_next_team"])


def add_win_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Did the possession team win? Ties score 0.5.

    `result` is home_score - away_score at the final whistle.
    """
    posteam_is_home = pl.col("posteam") == pl.col("home_team")
    margin = pl.when(posteam_is_home).then(pl.col("result")).otherwise(-pl.col("result"))

    return df.with_columns(
        [
            margin.alias("posteam_final_margin"),
            pl.when(margin > 0)
            .then(pl.lit(1.0))
            .when(margin < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.lit(0.5))
            .alias("posteam_won"),
        ]
    )


def add_sample_weights(df: pl.DataFrame) -> pl.DataFrame:
    """nflfastR-style training weights for the EP model.

    Two ideas, both about signal quality:

    1. A play many drives removed from the next score tells you less about the
       value of *this* state -- a lot of football happened in between. Weight
       decays with drive distance.
    2. Blowouts are played differently (prevent defense, running clock), so
       their plays are weaker evidence about normal-game state values. Weight
       decays with absolute score differential.

    Reproducing nflfastR's weighting keeps our EP numbers comparable to theirs
    in the benchmark, rather than differing for an uninteresting reason.
    """
    df = df.with_columns(
        (pl.col("_score_drive") - pl.col("drive")).abs().alias("_drive_dist")
    )

    drive_max = df["_drive_dist"].max()
    sd_max = df["score_differential"].abs().max()

    df = df.with_columns(
        [
            # Missing drive distance means "no score this half" -- treat as
            # maximally distant, i.e. lowest weight from this term.
            (
                1.0
                - (pl.col("_drive_dist").fill_null(drive_max) / max(drive_max, 1))
            ).alias("_w_drive"),
            (1.0 - (pl.col("score_differential").abs() / max(sd_max, 1))).alias(
                "_w_score"
            ),
        ]
    )

    # nflfastR sums the two components then rescales to [0, 1].
    total = pl.col("_w_drive") + pl.col("_w_score")
    df = df.with_columns(total.alias("_w_total"))
    lo, hi = df["_w_total"].min(), df["_w_total"].max()
    df = df.with_columns(
        ((pl.col("_w_total") - lo) / max(hi - lo, 1e-9)).alias("ep_sample_weight")
    )

    return df.drop(["_drive_dist", "_w_drive", "_w_score", "_w_total", "_score_drive"])


def add_second_half_kickoff(df: pl.DataFrame) -> pl.DataFrame:
    """Flag whether the possession team receives the second-half kickoff.

    Materially affects win probability late in the first half: trailing by 4
    with the ball and the second-half kickoff coming is a very different spot
    from trailing by 4 about to kick away.

    On a kickoff play nflverse sets posteam = kicking team, so the receiving
    team is `defteam`.
    """
    h2_receiver = (
        df.filter((pl.col("game_half") == "Half2") & (pl.col("kickoff_attempt") == 1))
        .sort(["game_id", "order_sequence", "play_id"])
        .group_by("game_id")
        .agg(pl.col("defteam").drop_nulls().first().alias("_h2_receiver"))
    )

    df = df.join(h2_receiver, on="game_id", how="left")
    return df.with_columns(
        (pl.col("posteam") == pl.col("_h2_receiver"))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("receive_2h_ko")
    ).drop("_h2_receiver")


def add_era_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Rule-change indicators, so the model can separate era effects.

    Within our 2016-2025 window the changes that plausibly move 4th-down math
    are the 17th game (more rest-related resting), the 2024 dynamic kickoff
    (changed the value of a punt/kickoff field-position swap), and the 2025
    regular-season overtime change.
    """
    season = pl.col("season")
    return df.with_columns(
        [
            ((season - 2016) / 9.0).alias("season_norm"),
            (season >= FIRST_17_GAME_SEASON).cast(pl.Int8).alias("is_17_game_era"),
            (season >= FIRST_DYNAMIC_KICKOFF_SEASON)
            .cast(pl.Int8)
            .alias("dynamic_kickoff_era"),
            (season >= FIRST_NEW_OT_SEASON).cast(pl.Int8).alias("ot_rules_2025"),
        ]
    )
