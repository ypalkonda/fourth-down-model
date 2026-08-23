"""Score every fourth down from 2016-2025 through the decision engine.

Run:  python scripts/score_history.py

Produces the dataset behind the blunder leaderboard and the coach
aggressiveness chart: for each real fourth down, what the coach did, what the
model would do, and the win probability that separates them.

Two deliberate choices about how the counterfactual is framed:

Team-neutral. The engine runs in league-average mode, so the recommendation
does not depend on which teams are playing. Team embeddings would let the
model say "punting was right because your offense was bad", which is a claim
about the roster rather than about the decision, and it would quietly excuse
coaches of weak teams on a leaderboard that is supposed to rank decisions.

League-average kicker. We do not know who would have kicked on plays where no
kick was attempted, so every field-goal branch uses the average leg. This
matters most for teams at the extremes of kicker quality; noted as a
limitation rather than papered over.

Fixed reference season. Every decision is scored as though it happened in
config.REFERENCE_SEASON, so coaches from different eras are graded on one
scale. See the comment on that constant for the measurement that forced it.
"""

from __future__ import annotations

import json
import time

import nflreadpy as nfl
import polars as pl

from fourthdown.config import (
    PROCESSED_DIR,
    REFERENCE_SEASON,
    REPORT_DIR,
    ensure_dirs,
)
from fourthdown.engine import ACTIONS, DecisionEngine, GameState, describe_yardline

# A coach needs a reasonable sample before a rate means anything.
MIN_COACH_DECISIONS = 150


def quarter_clock(game_seconds_remaining: float, qtr: int) -> str:
    """Time left in the quarter, formatted. `game_seconds_remaining` counts
    down the whole game, so displaying it directly reads as a quarter clock
    that says 34:19."""
    if qtr >= 5:
        return "OT"
    secs = max(0, int(game_seconds_remaining) - (4 - int(qtr)) * 900)
    return f"{secs // 60}:{secs % 60:02d}"


def build_state(r: dict, season: int = REFERENCE_SEASON) -> GameState:
    return GameState(
        yardline_100=float(r["yardline_100"]),
        ydstogo=float(r["ydstogo"]),
        down=4,
        game_seconds_remaining=float(r["game_seconds_remaining"]),
        half_seconds_remaining=float(r["half_seconds_remaining"]),
        score_differential=float(r["score_differential"]),
        posteam_timeouts_remaining=int(r["posteam_timeouts_remaining"] or 3),
        defteam_timeouts_remaining=int(r["defteam_timeouts_remaining"] or 3),
        posteam=r["posteam"],
        defteam=r["defteam"],
        is_home=int(r["posteam"] == r["home_team"]),
        roof=r["roof"] or "outdoors",
        receive_2h_ko=int(r["receive_2h_ko"] or 0),
        # Not r["season"]: see config.REFERENCE_SEASON.
        season=season,
        spread_line=float(r["posteam_spread_line"] or 0.0),
        wind=float(r["wind"] or 0.0),
        temp=float(r["temp"] or 60.0),
    )


def main() -> None:
    ensure_dirs()
    fd = pl.read_parquet(PROCESSED_DIR / "fourth_downs.parquet")

    # Attach head coaches so decisions can be attributed to a person.
    sch = nfl.load_schedules().select(["game_id", "home_coach", "away_coach"])
    fd = fd.join(sch, on="game_id", how="left").with_columns(
        pl.when(pl.col("posteam") == pl.col("home_team"))
        .then(pl.col("home_coach"))
        .otherwise(pl.col("away_coach"))
        .alias("coach")
    )
    print(f"scoring {fd.height:,} fourth downs (coach known for "
          f"{fd.filter(pl.col('coach').is_not_null()).height:,})")
    print(f"all decisions scored at reference season {REFERENCE_SEASON}, "
          f"team-neutral, league-average kicker")

    eng = DecisionEngine(league_average=True)
    rows = fd.to_dicts()
    out = []
    t0 = time.time()

    for i, r in enumerate(rows):
        if i and i % 10000 == 0:
            print(f"  {i:,}/{len(rows):,}  ({time.time()-t0:.0f}s)")
        res = eng.evaluate(build_state(r))
        actual = r["decision"]
        wp_best = max(res.wp.values())
        wp_actual = res.wp[actual]
        out.append(
            {
                "game_id": r["game_id"],
                "play_id": r["play_id"],
                "season": r["season"],
                "week": r["week"],
                "season_type": r["season_type"],
                "posteam": r["posteam"],
                "defteam": r["defteam"],
                "coach": r["coach"],
                "qtr": int(r["qtr"]),
                "game_seconds_remaining": r["game_seconds_remaining"],
                "clock": quarter_clock(r["game_seconds_remaining"], int(r["qtr"])),
                "yardline_100": r["yardline_100"],
                "ydstogo": r["ydstogo"],
                "score_differential": r["score_differential"],
                "field_position": describe_yardline(float(r["yardline_100"])),
                "actual": actual,
                "recommended": res.recommendation,
                "confidence": res.confidence,
                "is_tossup": int(res.confidence == "toss-up"),
                "wp_go": res.wp["go"],
                "wp_punt": res.wp["punt"],
                "wp_field_goal": res.wp["field_goal"],
                "ep_go": res.ep["go"],
                "ep_punt": res.ep["punt"],
                "ep_field_goal": res.ep["field_goal"],
                "convert_prob": res.convert_prob,
                "fg_make_prob": res.fg_make_prob,
                "wp_loss": max(0.0, wp_best - wp_actual),
                "agreed": int(actual == res.recommendation),
                "is_meaningful_decision": r["is_meaningful_decision"],
                "desc": (r["desc"] or "")[:200],
            }
        )

    scored = pl.DataFrame(out)
    scored.write_parquet(PROCESSED_DIR / "scored_fourth_downs.parquet")
    print(f"scored in {time.time()-t0:.0f}s -> scored_fourth_downs.parquet")

    # ------------------------------------------------------------------
    # A "real" decision: one that mattered and where the model has a view.
    real = scored.filter(
        (pl.col("is_meaningful_decision") == 1) & (pl.col("is_tossup") == 0)
    )

    print("\n" + "=" * 96)
    print("AGREEMENT WITH COACHES")
    print("=" * 96)
    print(f"  meaningful, non-tossup decisions : {real.height:,} of {scored.height:,}")
    print(f"  model agrees with coach          : {100*real['agreed'].mean():.1f}%")
    print(f"  mean WP lost per decision        : {100*real['wp_loss'].mean():.2f} pts")

    print("\n  agreement by what the model recommends")
    for a in ACTIONS:
        s = real.filter(pl.col("recommended") == a)
        if s.height:
            print(f"    model says {a:<11} n={s.height:>6,}  coach agreed "
                  f"{100*s['agreed'].mean():5.1f}%  mean WP lost "
                  f"{100*s['wp_loss'].mean():4.2f}")

    print("\n  the asymmetry: what coaches do when the model says GO")
    sg = real.filter(pl.col("recommended") == "go")
    for a in ACTIONS:
        n = sg.filter(pl.col("actual") == a).height
        print(f"    actually {a:<11} {n:>6,}  ({100*n/max(sg.height,1):5.1f}%)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("BIGGEST BLUNDERS 2016-2025 (single decisions, by win probability given away)")
    print("=" * 96)
    top = real.sort("wp_loss", descending=True).head(15)
    for r in top.to_dicts():
        print(
            f"  {100*r['wp_loss']:5.1f}pp  {r['season']} wk{r['week']:<2} "
            f"{r['posteam']} v {r['defteam']:<4} Q{r['qtr']} {r['clock']:>5} "
            f"| 4th & {r['ydstogo']:<2.0f} {r['field_position']:<8} "
            f"| {r['score_differential']:+3.0f} "
            f"| did {r['actual']:<10} should {r['recommended']:<10} | {r['coach']}"
        )

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"COACH AGGRESSIVENESS (min {MIN_COACH_DECISIONS} meaningful decisions)")
    print("=" * 96)
    coaches = (
        real.filter(pl.col("coach").is_not_null())
        .group_by("coach")
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("actual") == "go").mean().alias("actual_go_rate"),
                (pl.col("recommended") == "go").mean().alias("optimal_go_rate"),
                pl.col("wp_loss").mean().alias("mean_wp_loss"),
                pl.col("wp_loss").sum().alias("total_wp_loss"),
                pl.col("agreed").mean().alias("agree_rate"),
            ]
        )
        .filter(pl.col("n") >= MIN_COACH_DECISIONS)
        .with_columns(
            (pl.col("actual_go_rate") / pl.col("optimal_go_rate")).alias("aggression_index")
        )
        .sort("mean_wp_loss")
    )

    print(f"  {'coach':<24}{'n':>6}{'go%':>7}{'opt go%':>9}{'index':>7}"
          f"{'agree%':>8}{'WP lost/dec':>13}")
    print("  " + "-" * 74)
    for r in coaches.head(12).to_dicts():
        print(f"  {r['coach']:<24}{r['n']:>6,}{100*r['actual_go_rate']:>7.1f}"
              f"{100*r['optimal_go_rate']:>9.1f}{r['aggression_index']:>7.2f}"
              f"{100*r['agree_rate']:>8.1f}{100*r['mean_wp_loss']:>13.2f}")
    print("  " + "." * 74)
    for r in coaches.tail(8).to_dicts():
        print(f"  {r['coach']:<24}{r['n']:>6,}{100*r['actual_go_rate']:>7.1f}"
              f"{100*r['optimal_go_rate']:>9.1f}{r['aggression_index']:>7.2f}"
              f"{100*r['agree_rate']:>8.1f}{100*r['mean_wp_loss']:>13.2f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("LEAGUE TREND: are coaches catching up to the model?")
    print("=" * 96)
    seasons = (
        real.group_by("season")
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("actual") == "go").mean().alias("actual_go_rate"),
                (pl.col("recommended") == "go").mean().alias("optimal_go_rate"),
                pl.col("wp_loss").mean().alias("mean_wp_loss"),
            ]
        )
        .sort("season")
    )
    print(f"  {'season':<9}{'n':>7}{'actual go%':>13}{'optimal go%':>14}{'WP lost/dec':>14}")
    for r in seasons.to_dicts():
        print(f"  {r['season']:<9}{r['n']:>7,}{100*r['actual_go_rate']:>13.1f}"
              f"{100*r['optimal_go_rate']:>14.1f}{100*r['mean_wp_loss']:>14.2f}")

    coaches.write_parquet(PROCESSED_DIR / "coach_summary.parquet")
    seasons.write_parquet(PROCESSED_DIR / "season_summary.parquet")
    (REPORT_DIR / "history_summary.json").write_text(
        json.dumps(
            {
                "n_scored": scored.height,
                "n_real_decisions": real.height,
                "agreement_rate": float(real["agreed"].mean()),
                "mean_wp_loss": float(real["wp_loss"].mean()),
                "seasons": seasons.to_dicts(),
            },
            indent=2,
        )
    )
    print(f"\nwrote coach_summary.parquet, season_summary.parquet, history_summary.json")


if __name__ == "__main__":
    main()
