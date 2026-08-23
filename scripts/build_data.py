"""Build every processed dataset from raw nflverse play-by-play.

Run:  python scripts/build_data.py

Prints a validation block at the end. The empirical-EP-vs-nflfastR comparison
is the load-bearing check: if our next-score labels were wrong, that
correlation would fall apart.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fourthdown.config import (
    NEXT_SCORE_CLASSES,
    NEXT_SCORE_VALUES,
    PROCESSED_DIR,
    SEASONS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    ensure_dirs,
)
from fourthdown.features import (
    build_conversion_dataset,
    build_fg_dataset,
    build_fourth_down_dataset,
    build_punt_dataset,
    build_state_dataset,
    prepare,
)
from fourthdown.ingest import load_pbp

POINT_MAP = dict(zip(NEXT_SCORE_CLASSES, NEXT_SCORE_VALUES))


def main() -> None:
    ensure_dirs()

    print(f"Loading play-by-play for {SEASONS[0]}-{SEASONS[-1]} ...")
    raw = load_pbp(SEASONS)
    print(f"  raw plays: {raw.height:,}")

    print("Building labels and features ...")
    df = prepare(raw)

    state = build_state_dataset(df)
    fourth = build_fourth_down_dataset(df)
    conv = build_conversion_dataset(df)
    punts = build_punt_dataset(df)
    fgs = build_fg_dataset(df)

    state.write_parquet(PROCESSED_DIR / "state_plays.parquet")
    fourth.write_parquet(PROCESSED_DIR / "fourth_downs.parquet")
    conv.write_parquet(PROCESSED_DIR / "conversions.parquet")
    punts.write_parquet(PROCESSED_DIR / "punts.parquet")
    fgs.write_parquet(PROCESSED_DIR / "field_goals.parquet")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("VALIDATION")
    print("=" * 66)

    print(f"\nstate_plays   : {state.height:,} rows  ({len(state.columns)} cols)")
    print(f"fourth_downs  : {fourth.height:,} rows")
    print(f"conversions   : {conv.height:,} rows")
    print(f"punts         : {punts.height:,} rows")
    print(f"field_goals   : {fgs.height:,} rows")

    print("\n-- punt outcomes: where the receiving team starts --")
    pv = punts.filter(pl.col("return_td") == 0)
    print(
        f"   mean start (own yardline)  {100 - pv['defense_start_yardline'].mean():.1f}"
        f"   |  touchbacks {100*(pv['defense_start_yardline'] == 80).mean():.1f}%"
        f"   |  return TDs {100*punts['return_td'].mean():.2f}%"
        f"   |  blocked {100*punts['blocked'].mean():.2f}%"
    )
    pb = (
        punts.filter((pl.col("return_td") == 0) & pl.col("yardline_100").is_between(40, 80))
        .with_columns((pl.col("yardline_100") // 10 * 10).alias("los_bin"))
        .group_by("los_bin")
        .agg([pl.len().alias("n"),
              (100 - pl.col("defense_start_yardline")).mean().alias("opp_start"),
              (pl.col("yardline_100") - (100 - pl.col("defense_start_yardline"))).mean().alias("net")])
        .sort("los_bin")
    )
    print(f"   {'punt from own':<16}{'n':>7}{'opp starts at':>15}{'net yds':>10}")
    for r in pb.to_dicts():
        print(f"   own {int(100-r['los_bin'])-9}-{int(100-r['los_bin']):<8}{r['n']:>7,}"
              f"{r['opp_start']:>15.1f}{r['net']:>10.1f}")

    print("\n-- field goals by distance --")
    fb = (
        fgs.with_columns((pl.col("fg_distance") // 5 * 5).alias("d_bin"))
        .group_by("d_bin")
        .agg([pl.len().alias("n"), pl.col("fg_made").mean().alias("pct")])
        .sort("d_bin")
        .filter(pl.col("n") >= 20)
    )
    for r in fb.to_dicts():
        bar = "#" * int(r["pct"] * 40)
        print(f"   {int(r['d_bin'])}-{int(r['d_bin'])+4:<3} yd  n={r['n']:>5,}  {r['pct']*100:5.1f}%  {bar}")

    print("\n-- rows per season (state) --")
    per = state.group_by("season").len().sort("season")
    for r in per.to_dicts():
        print(f"   {r['season']}  {r['len']:>7,}")

    print("\n-- next_score_type distribution --")
    dist = state.group_by("next_score_type").len().sort("len", descending=True)
    for r in dist.to_dicts():
        pct = 100 * r["len"] / state.height
        print(f"   {r['next_score_type']:<16} {r['len']:>7,}  {pct:5.2f}%")

    # The real test of the labels. Bin by field position and compare our
    # empirical expected points against nflfastR's model output. These are
    # computed completely independently -- ours straight from next-score
    # outcomes, theirs from a fitted XGBoost model -- so agreement means our
    # label construction reproduces the standard.
    print("\n-- empirical EP from our labels vs nflfastR `ep` (1st & 10) --")
    first_downs = state.filter((pl.col("down") == 1) & (pl.col("ydstogo") == 10))
    binned = (
        first_downs.with_columns(
            [
                (pl.col("yardline_100") // 10 * 10).alias("yl_bin"),
                pl.col("next_score_type")
                .replace_strict(POINT_MAP, default=0.0)
                .alias("pts"),
            ]
        )
        .group_by("yl_bin")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("pts").mean().alias("empirical_ep"),
                pl.col("ep").mean().alias("nflfastr_ep"),
            ]
        )
        .sort("yl_bin")
    )
    print(f"   {'yards to EZ':<14}{'n':>8}{'ours':>9}{'nflfastR':>11}{'diff':>8}")
    for r in binned.to_dicts():
        d = r["empirical_ep"] - r["nflfastr_ep"]
        print(
            f"   {int(r['yl_bin']):>3}-{int(r['yl_bin'])+9:<10}{r['n']:>8,}"
            f"{r['empirical_ep']:>9.2f}{r['nflfastr_ep']:>11.2f}{d:>8.2f}"
        )
    corr = np.corrcoef(binned["empirical_ep"], binned["nflfastr_ep"])[0, 1]
    mae = float((binned["empirical_ep"] - binned["nflfastr_ep"]).abs().mean())
    print(f"   correlation = {corr:.4f}   mean abs diff = {mae:.3f} pts")

    print("\n-- fourth-down decisions --")
    dec = fourth.group_by("decision").len().sort("len", descending=True)
    for r in dec.to_dicts():
        pct = 100 * r["len"] / fourth.height
        print(f"   {r['decision']:<14} {r['len']:>7,}  {pct:5.2f}%")
    inferred = fourth["decision_inferred"].sum()
    print(f"   (intent recovered from description for {inferred:,} penalty-nullified plays)")

    meaningful = int(fourth["is_meaningful_decision"].sum())
    print(f"   meaningful decisions: {meaningful:,} / {fourth.height:,}")

    print("\n-- go-rate by season (the aggressiveness trend) --")
    go = (
        fourth.filter(pl.col("is_meaningful_decision") == 1)
        .group_by("season")
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("decision") == "go").mean().alias("go_rate"),
            ]
        )
        .sort("season")
    )
    for r in go.to_dicts():
        bar = "#" * int(r["go_rate"] * 100)
        print(f"   {r['season']}  {r['go_rate']*100:5.1f}%  {bar}")

    print("\n-- conversion rate by distance (3rd & 4th down pooled) --")
    cv = (
        conv.filter(pl.col("ydstogo") <= 10)
        .group_by("ydstogo")
        .agg([pl.len().alias("n"), pl.col("converted").mean().alias("rate")])
        .sort("ydstogo")
    )
    for r in cv.to_dicts():
        print(f"   4th/3rd & {int(r['ydstogo']):<3} n={r['n']:>7,}  {r['rate']*100:5.1f}%")

    print("\n-- split sizes --")
    for name, seasons in [
        ("train", TRAIN_SEASONS),
        ("val", VAL_SEASONS),
        ("test", TEST_SEASONS),
    ]:
        n = state.filter(pl.col("season").is_in(seasons)).height
        print(f"   {name:<6} {seasons[0]}-{seasons[-1]}  {n:>8,} plays")

    print("\nWrote parquet files to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
