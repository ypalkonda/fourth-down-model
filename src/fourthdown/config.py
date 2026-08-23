"""Central configuration: seasons, splits, feature sets, scoring values.

Everything that a reviewer might want to challenge ("why these seasons?",
"why this split?") lives here rather than being scattered through the code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------
# Ten completed seasons. 2025 finished in Feb 2026; 2026 has not been played.
SEASONS = list(range(2016, 2026))

# Temporal split, deliberately NOT random.
#
# A random play-level split leaks the target badly: every play in a drive
# shares the same next-score label, so the same outcome would appear on both
# sides of the split. Splitting by season also matches how the model would
# actually be used -- fit on the past, predict the future.
TRAIN_SEASONS = list(range(2016, 2024))  # 2016-2023
VAL_SEASONS = [2024]
TEST_SEASONS = [2025]

# --------------------------------------------------------------------------
# Next-score classes (the EP model's target)
# --------------------------------------------------------------------------
# Point values match nflfastR's EP model exactly, so our expected-point
# numbers are directly comparable against the field-standard implementation
# in the validation report. TD is valued at 7 (i.e. the PAT is folded in
# rather than modelled separately).
NEXT_SCORE_CLASSES = [
    "Touchdown",
    "Field_Goal",
    "Safety",
    "No_Score",
    "Opp_Safety",
    "Opp_Field_Goal",
    "Opp_Touchdown",
]
NEXT_SCORE_VALUES = [7.0, 3.0, 2.0, 0.0, -2.0, -3.0, -7.0]
NEXT_SCORE_INDEX = {c: i for i, c in enumerate(NEXT_SCORE_CLASSES)}

# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
# Features for the shared trunk.
#
# Deliberately score-AGNOSTIC. Expected points answers "what is this field
# position and down worth?", which is a property of the state, not of the
# scoreboard. Feeding score differential in here would conflate state value
# with game script (trailing teams pass more, leading teams run out the
# clock) and would also break comparability with nflfastR, whose EP model
# excludes score for the same reason. Score enters at the WP head instead.
TRUNK_FEATURES = [
    "yardline_100",              # distance to opponent end zone (1-99)
    "ydstogo",                   # yards needed for a first down
    "down",                      # 1-4
    "goal_to_go",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",                   # posteam is the home team
    "roof_dome",
    "roof_outdoors",
    "roof_retractable",
    "receive_2h_ko",             # posteam receives the 2nd-half kickoff
    "season_norm",               # linear season index, captures league drift
    "is_17_game_era",            # 2021+
    "dynamic_kickoff_era",       # 2024+
    "ot_rules_2025",             # 2025+
]

# Score-dependent features, concatenated onto the trunk output inside the WP
# head only. Time and score interact non-linearly -- a 3-point lead is worth
# far more with 30 seconds left than with 30 minutes left -- so we hand the
# model explicit interaction terms rather than hoping it discovers them.
WP_HEAD_FEATURES = [
    "score_differential",        # posteam - defteam
    "diff_time_ratio",           # score_diff amplified as the clock runs down
    "spread_time",               # pregame spread, decayed toward zero
    "posteam_spread_line",       # market view of team strength, posteam POV
]

# Everything the state dataset needs to carry.
STATE_FEATURES = TRUNK_FEATURES + WP_HEAD_FEATURES

# --------------------------------------------------------------------------
# Rule-change boundaries (documented so the era flags are auditable)
# --------------------------------------------------------------------------
FIRST_17_GAME_SEASON = 2021
FIRST_DYNAMIC_KICKOFF_SEASON = 2024
FIRST_NEW_OT_SEASON = 2025

# Season used when scoring historical decisions retrospectively.
#
# The trunk carries era features (`season_norm`, `is_17_game_era`) and they
# move recommendations more than they should: re-scoring one fixed set of 900
# real 2019 fourth downs under different season labels swings the share the
# model would go for from 18.6% to 30.7%, while the underlying situation mix
# across those seasons barely moves (mean yards-to-go 7.46-8.09). Most of that
# swing is the model reading era markers, not football.
#
# Letting it through would make the leaderboard incoherent -- a 2016 coach and
# a 2025 coach graded on different scales -- and would fake a "the optimal go
# rate is rising" trend that is really just the season feature. Scoring every
# historical decision at one reference season asks the question we actually
# care about: by a single consistent standard, was this call right?
REFERENCE_SEASON = 2025

# --------------------------------------------------------------------------
# Outcome distributions
# --------------------------------------------------------------------------
# Yards-gained bins for the "go for it" model.
#
# Resolution is deliberately uneven. Whether a fourth-down attempt gains 1 or
# 2 yards decides the game; whether it gains 47 or 52 barely does. So every
# integer from -5 to +20 gets its own bin and the tails are coarse. Modelling
# the full distribution rather than a convert/fail coin flip matters because
# the failure spot varies -- a stuffed run at the line and a 9-yard sack on
# 4th & 12 hand the opponent very different field position.
GAIN_BIN_LOWER = [-99, *range(-5, 21), 21, 26, 31, 41, 61]
GAIN_BIN_UPPER = [-6, *range(-5, 21), 25, 30, 40, 60, 99]
GAIN_REPRESENTATIVE = [-8, *range(-5, 21), 23, 28, 35, 50, 75]
N_GAIN_BINS = len(GAIN_BIN_LOWER)

# Punt outcome bins: where the receiving team starts, in its own frame, in
# 5-yard buckets, plus one absorbing outcome for a return touchdown.
PUNT_BIN_WIDTH = 5
N_PUNT_FIELD_BINS = 100 // PUNT_BIN_WIDTH
PUNT_RETURN_TD_BIN = N_PUNT_FIELD_BINS
N_PUNT_BINS = N_PUNT_FIELD_BINS + 1

# --------------------------------------------------------------------------
# Field-goal geometry
# --------------------------------------------------------------------------
# Kick distance = yards to the end zone + 10 (end zone) + 7 (snap depth).
FG_SNAP_DEPTH = 7
END_ZONE_DEPTH = 10


def fg_distance_from_yardline(yardline_100: float) -> float:
    """Attempted FG distance in yards from `yardline_100`."""
    return yardline_100 + END_ZONE_DEPTH + FG_SNAP_DEPTH


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
