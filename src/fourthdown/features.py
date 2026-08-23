"""Feature engineering and dataset assembly.

Three datasets come out of here:

* `build_state_dataset`  -- every scrimmage play, used to fit EP and WP.
  This is the important one to understand: the state-value models train on all
  ~350k regular-season scrimmage plays, NOT just fourth downs. Fourth downs
  are where the model is *applied*, not where it learns what a game state is
  worth.

* `build_fourth_down_dataset` -- the ~35k fourth downs, with the coach's
  actual decision recovered, for evaluation and the dashboard.

* `build_conversion_dataset` -- third and fourth down rushes/passes, used to
  fit P(convert) and the yards-gained distribution.
"""

from __future__ import annotations

import polars as pl

from .config import STATE_FEATURES
from .labels import (
    add_era_flags,
    add_next_score_labels,
    add_sample_weights,
    add_second_half_kickoff,
    add_win_labels,
)

# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Roof one-hots, home flag, and the time/score interaction terms."""
    elapsed_share = (3600.0 - pl.col("game_seconds_remaining")) / 3600.0
    # Exponential decay: the same point differential is worth progressively
    # more as the clock runs down. nflfastR uses the same -4 coefficient, so
    # our WP model and theirs are comparable feature-for-feature.
    decay = (-4.0 * elapsed_share).exp()

    # `spread_line` is stated from the home team's perspective; flip it when
    # the possession team is the road team.
    posteam_spread = (
        pl.when(pl.col("posteam") == pl.col("home_team"))
        .then(pl.col("spread_line"))
        .otherwise(-pl.col("spread_line"))
    )

    return df.with_columns(
        [
            (pl.col("posteam") == pl.col("home_team")).cast(pl.Int8).alias("is_home"),
            (pl.col("roof") == "dome").cast(pl.Int8).alias("roof_dome"),
            (pl.col("roof").is_in(["outdoors", "open"])).cast(pl.Int8).alias("roof_outdoors"),
            (pl.col("roof").is_in(["closed", "retractable"]))
            .cast(pl.Int8)
            .alias("roof_retractable"),
            pl.col("goal_to_go").fill_null(0).cast(pl.Int8).alias("goal_to_go"),
            (pl.col("score_differential") / decay).alias("diff_time_ratio"),
            (posteam_spread * decay).alias("spread_time"),
            posteam_spread.alias("posteam_spread_line"),
        ]
    )


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    """Run the full label + feature chain over raw pbp."""
    df = add_second_half_kickoff(df)
    df = add_next_score_labels(df)
    df = add_win_labels(df)
    df = add_sample_weights(df)
    df = add_era_flags(df)
    df = add_derived_features(df)
    return df


# --------------------------------------------------------------------------
# State-value dataset (EP / WP training)
# --------------------------------------------------------------------------

REQUIRED_STATE_COLS = [
    "posteam",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "next_score_type",
    "posteam_won",
]


def build_state_dataset(df: pl.DataFrame, regular_season_only: bool = True) -> pl.DataFrame:
    """Every well-formed scrimmage play, with EP and WP targets attached.

    Kneels and spikes are dropped. They are clock-management artifacts, not
    attempts to gain yards, and leaving them in biases EP downward in exactly
    the end-of-half spots where fourth-down decisions get interesting.
    """
    out = df.filter(pl.col("down").is_not_null())

    if regular_season_only:
        out = out.filter(pl.col("season_type") == "REG")

    out = out.filter(
        (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
    )

    for col in REQUIRED_STATE_COLS:
        out = out.filter(pl.col(col).is_not_null())

    # Conversion target for the third head. Only defined on third/fourth-down
    # rush/pass attempts; null everywhere else, and the training loop masks
    # the loss accordingly so punts and first downs contribute nothing to it.
    is_conv_attempt = (
        pl.col("down").is_in([3, 4])
        & pl.col("play_type").is_in(["run", "pass"])
        & (pl.col("penalty") != 1)
        & pl.col("yards_gained").is_not_null()
    )
    out = out.with_columns(
        [
            pl.when(is_conv_attempt)
            .then((pl.col("yards_gained") >= pl.col("ydstogo")).cast(pl.Int8))
            .otherwise(None)
            .alias("converted"),
            is_conv_attempt.cast(pl.Int8).alias("convert_mask"),
        ]
    )

    keep = (
        ["game_id", "play_id", "season", "week", "posteam", "defteam", "home_team"]
        + STATE_FEATURES
        + [
            "next_score_type",
            "posteam_won",
            "posteam_final_margin",
            "converted",
            "convert_mask",
            "yards_gained",
            "ep_sample_weight",
            "ep",   # nflfastR benchmark only -- never a model input
            "wp",
            "vegas_wp",
        ]
    )
    return out.select([c for c in keep if c in out.columns])


# --------------------------------------------------------------------------
# Fourth-down decision dataset
# --------------------------------------------------------------------------

# Penalty-nullified fourth downs ("no_play") still reveal the coach's intent
# in the play description -- the punt was struck, the kick was snapped, the
# offense broke the huddle. We recover intent by pattern rather than throwing
# away ~6% of the sample, and flag the rows we inferred.
_PUNT_PAT = r"(?i)\bpunts?\b|\bpunt is blocked\b"
_FG_PAT = r"(?i)field goal|\bFG\b"
_GO_PAT = r"(?i)pass|scramble|left end|right end|up the middle|left guard|right guard|left tackle|right tackle|sacked|kneel"


def classify_decision(df: pl.DataFrame) -> pl.DataFrame:
    """Map each fourth down to go / punt / field_goal."""
    pt = pl.col("play_type")
    desc = pl.col("desc").fill_null("")

    explicit = (
        pl.when(pt == "punt")
        .then(pl.lit("punt"))
        .when(pt == "field_goal")
        .then(pl.lit("field_goal"))
        .when(pt.is_in(["run", "pass"]))
        .then(pl.lit("go"))
        .otherwise(None)
    )

    inferred = (
        pl.when(desc.str.contains(_PUNT_PAT))
        .then(pl.lit("punt"))
        .when(desc.str.contains(_FG_PAT))
        .then(pl.lit("field_goal"))
        .when(desc.str.contains(_GO_PAT))
        .then(pl.lit("go"))
        .otherwise(None)
    )

    return df.with_columns(
        [
            explicit.fill_null(inferred).alias("decision"),
            explicit.is_null().cast(pl.Int8).alias("decision_inferred"),
        ]
    )


def build_fourth_down_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Fourth downs with the coach's actual call, for eval and the dashboard."""
    out = df.filter(
        (pl.col("down") == 4)
        & pl.col("posteam").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("ydstogo").is_not_null()
    )
    out = classify_decision(out)
    out = out.filter(pl.col("decision").is_not_null())

    # Situations where the "decision" isn't really a decision. Flagged rather
    # than dropped so the dashboard can exclude them from blunder rankings
    # without silently shrinking the dataset.
    desperation = (
        (pl.col("qtr") >= 4)
        & (pl.col("score_differential") < 0)
        & (pl.col("game_seconds_remaining") < 300)
    )
    end_of_half = pl.col("half_seconds_remaining") < 30

    return out.with_columns(
        [
            desperation.cast(pl.Int8).alias("is_desperation"),
            end_of_half.cast(pl.Int8).alias("is_end_of_half"),
            (~(desperation | end_of_half)).cast(pl.Int8).alias("is_meaningful_decision"),
        ]
    )


# --------------------------------------------------------------------------
# Where the ball actually ended up
# --------------------------------------------------------------------------


def add_next_scrimmage_state(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the field position and possessing team of the NEXT scrimmage play.

    This is how we learn what a punt or a missed field goal is actually worth.
    The alternative -- reconstructing the spot arithmetically from
    `kick_distance` and `return_yards` -- has to special-case touchbacks,
    fair catches, out-of-bounds kicks, muffs, blocks, and penalties, and gets
    at least one of them wrong. Reading the next snap's `yardline_100` is
    ground truth for all of them simultaneously.

    Indexing is done over scrimmage plays only, so kickoffs, PATs, timeouts
    and end-of-quarter markers cannot land between a punt and the play we
    attribute to it.
    """
    scrim = df.filter(pl.col("down").is_not_null()).sort(
        ["game_id", "order_sequence", "play_id"]
    )
    scrim = scrim.with_columns(pl.int_range(pl.len()).over("game_id").alias("_si"))

    nxt = scrim.select(
        [
            "game_id",
            (pl.col("_si") - 1).alias("_si"),
            pl.col("posteam").alias("next_posteam"),
            pl.col("yardline_100").alias("next_yardline_100"),
            pl.col("game_half").alias("next_game_half"),
            pl.col("game_seconds_remaining").alias("next_game_seconds_remaining"),
        ]
    )
    return scrim.join(nxt, on=["game_id", "_si"], how="left").drop("_si")


def _resulting_defense_yardline() -> pl.Expr:
    """Field position of the team that takes over, from ITS OWN perspective.

    `yardline_100` is always measured toward the end zone the possessing team
    is attacking, so when possession flips the number has to be mirrored.
    """
    return (
        pl.when(pl.col("next_posteam") == pl.col("posteam"))
        # Possession did not change (e.g. a blocked punt recovered by the
        # kicking team). Mirror it so the column always means "the opponent's
        # starting spot"; these rows are rare and filtered out downstream.
        .then(100 - pl.col("next_yardline_100"))
        .otherwise(pl.col("next_yardline_100"))
    )


# --------------------------------------------------------------------------
# Punt dataset
# --------------------------------------------------------------------------


def build_punt_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Punts, labelled with where the receiving team actually started.

    A punt is only meaningful as a distribution. Netting 45 yards to the
    opponent's 12 and netting 45 yards into the end zone for a touchback are
    the same average and very different decisions, and the difference is
    exactly what a fourth-down model near midfield turns on.
    """
    out = add_next_scrimmage_state(df).filter(
        (pl.col("punt_attempt") == 1)
        & pl.col("posteam").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )

    out = out.with_columns(
        [
            _resulting_defense_yardline().alias("defense_start_yardline"),
            # A punt returned for a touchdown is rare (~0.3%) but is the worst
            # realistic outcome, so it gets its own branch instead of being
            # averaged into the field-position distribution.
            (
                (pl.col("touchdown") == 1) & (pl.col("td_team") != pl.col("posteam"))
            ).cast(pl.Int8).alias("return_td"),
            (pl.col("punt_blocked") == 1).cast(pl.Int8).alias("blocked"),
        ]
    )

    # Keep rows where we either saw the next snap or the punt ended in a
    # return touchdown (which has no following scrimmage play for us to read).
    return out.filter(
        (pl.col("return_td") == 1) | pl.col("defense_start_yardline").is_not_null()
    )


# --------------------------------------------------------------------------
# Field-goal dataset
# --------------------------------------------------------------------------

MIN_KICKER_ATTEMPTS = 25


def build_fg_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Field-goal attempts with kick distance, conditions, and kicker."""
    out = df.filter(
        (pl.col("field_goal_attempt") == 1)
        & pl.col("field_goal_result").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )

    return out.with_columns(
        [
            (pl.col("field_goal_result") == "made").cast(pl.Int8).alias("fg_made"),
            # Prefer the recorded kick distance; fall back to the geometry
            # (yards to the end zone + 10 end zone + 7 snap depth).
            pl.col("kick_distance")
            .fill_null(pl.col("yardline_100") + 17)
            .alias("fg_distance"),
            pl.col("temp").fill_null(60.0).alias("temp_filled"),
            pl.col("wind").fill_null(0.0).alias("wind_filled"),
            # Indoors, weather is irrelevant; the model should not read a
            # missing temperature as "60 degrees and calm outdoors".
            (pl.col("roof").is_in(["dome", "closed"])).cast(pl.Int8).alias("is_indoor"),
        ]
    )


# --------------------------------------------------------------------------
# Conversion dataset (P(convert) + yards-gained distribution)
# --------------------------------------------------------------------------


def build_conversion_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Rush/pass attempts on third and fourth down.

    Fourth-down go attempts alone are sparse (~760 a season) and badly
    selected -- coaches mostly go for it on short yardage or when desperate,
    so the sample under-represents the situations we most need to predict.
    Third downs are the same football problem (convert or turn it over) with
    ~15x the volume, so we pool them and give the model a `down` feature to
    learn any residual difference.
    """
    out = df.filter(
        pl.col("down").is_in([3, 4])
        & (pl.col("play_type").is_in(["run", "pass"]))
        & (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
        & (pl.col("penalty") != 1)          # nullified plays have no clean yardage
        & pl.col("yards_gained").is_not_null()
        & pl.col("posteam").is_not_null()
    )

    return out.with_columns(
        [
            (pl.col("yards_gained") >= pl.col("ydstogo")).cast(pl.Int8).alias("converted"),
            # Clip to the field: you cannot gain more than the distance to the
            # end zone, and losses are floored well short of a safety.
            pl.col("yards_gained")
            .clip(-20, 100)
            .alias("yards_gained_clipped"),
        ]
    )
