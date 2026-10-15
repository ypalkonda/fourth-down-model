"""Flask API over the decision engine and the scored history.

Run:  make api        (http://127.0.0.1:5001)

The engine and the scored parquet files are loaded once at import and held in
memory -- the models are a few hundred KB and the history is 41k rows, so
there is no reason to touch disk per request.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

# Make src/ importable without an install step (Vercel, gunicorn, plain python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from fourthdown.config import PROCESSED_DIR, REFERENCE_SEASON
from fourthdown.engine import (
    LEAN_MARGIN,
    TOSSUP_MARGIN,
    DecisionEngine,
    GameState,
    describe_yardline,
)

DIST_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "dist"

app = Flask(__name__, static_folder=None)
CORS(app)

ENGINE = DecisionEngine(league_average=True)
SCORED = pl.read_parquet(PROCESSED_DIR / "scored_fourth_downs.parquet")
COACHES = pl.read_parquet(PROCESSED_DIR / "coach_summary.parquet")
SEASONS = pl.read_parquet(PROCESSED_DIR / "season_summary.parquet")
REAL = SCORED.filter(
    (pl.col("is_meaningful_decision") == 1) & (pl.col("is_tossup") == 0)
)


# --------------------------------------------------------------------------
def _clean(v, default=0.0):
    """JSON cannot carry NaN/Inf; Polars nulls arrive as None."""
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return v


def state_from_payload(p: dict) -> GameState:
    qtr = int(p.get("qtr", 4))
    clock = float(p.get("clock_seconds", 900))  # seconds left in the quarter
    game_left = clock + (4 - min(qtr, 4)) * 900
    half_left = clock + (900 if qtr in (1, 3) else 0)

    return GameState(
        yardline_100=float(p.get("yardline_100", 50)),
        ydstogo=float(p.get("ydstogo", 4)),
        down=4,
        game_seconds_remaining=game_left,
        half_seconds_remaining=half_left,
        score_differential=float(p.get("score_differential", 0)),
        posteam_timeouts_remaining=int(p.get("posteam_timeouts", 3)),
        defteam_timeouts_remaining=int(p.get("defteam_timeouts", 3)),
        is_home=int(p.get("is_home", 1)),
        roof=p.get("roof", "outdoors"),
        receive_2h_ko=int(p.get("receive_2h_ko", 0)),
        season=REFERENCE_SEASON,
        spread_line=float(p.get("spread_line", 0)),
        wind=float(p.get("wind", 0)),
        temp=float(p.get("temp", 60)),
    )


def result_json(s: GameState, r) -> dict:
    return {
        "recommendation": r.recommendation,
        "confidence": r.confidence,
        "wp_margin": r.wp_margin,
        "wp": r.wp,
        "ep": r.ep,
        "convert_prob": r.convert_prob,
        "fg_make_prob": r.fg_make_prob,
        "fg_distance": s.yardline_100 + 17,
        "field_position": describe_yardline(s.yardline_100),
        "branches": r.branches,
    }


# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "scored_rows": SCORED.height})


@app.get("/api/meta")
def meta():
    return jsonify(
        {
            "reference_season": REFERENCE_SEASON,
            "tossup_margin": TOSSUP_MARGIN,
            "lean_margin": LEAN_MARGIN,
            "seasons": sorted(SCORED["season"].unique().to_list()),
            "teams": sorted(SCORED["posteam"].drop_nulls().unique().to_list()),
            "coaches": sorted(COACHES["coach"].to_list()),
            "totals": {
                "scored": SCORED.height,
                "real_decisions": REAL.height,
                "tossups": int((SCORED["is_tossup"] == 1).sum()),
                "agreement": float(REAL["agreed"].mean()),
                "mean_wp_loss": float(REAL["wp_loss"].mean()),
            },
            "by_recommendation": [
                {
                    "action": a,
                    "n": REAL.filter(pl.col("recommended") == a).height,
                    "agreement": float(
                        REAL.filter(pl.col("recommended") == a)["agreed"].mean()
                    ),
                    "mean_wp_loss": float(
                        REAL.filter(pl.col("recommended") == a)["wp_loss"].mean()
                    ),
                }
                for a in ("go", "punt", "field_goal")
            ],
        }
    )


@app.post("/api/evaluate")
def evaluate():
    p = request.get_json(force=True) or {}
    s = state_from_payload(p)
    return jsonify(result_json(s, ENGINE.evaluate(s)))


# The grid is the expensive endpoint: a few hundred situations per call. It is
# also highly repeatable as the user drags one slider, so it is memoized on the
# exact parameter tuple.
@lru_cache(maxsize=256)
def _grid(score_diff, clock, qtr, po_to, de_to, roof, is_home, spread, max_togo, step):
    togos = list(range(1, max_togo + 1))
    yards = list(range(2, 100, step))
    combos = [(t, y) for t in togos for y in yards if t <= y]

    states = [
        state_from_payload(
            {
                "yardline_100": y,
                "ydstogo": t,
                "score_differential": score_diff,
                "clock_seconds": clock,
                "qtr": qtr,
                "posteam_timeouts": po_to,
                "defteam_timeouts": de_to,
                "roof": roof,
                "is_home": is_home,
                "spread_line": spread,
            }
        )
        for t, y in combos
    ]
    results = ENGINE.evaluate_many(states)

    return {
        "togos": togos,
        "yards": yards,
        "cells": [
            {
                "togo": t,
                "yardline_100": y,
                "rec": r.recommendation,
                "confidence": r.confidence,
                "margin": round(r.wp_margin, 4),
                "wp": {k: round(v, 4) for k, v in r.wp.items()},
            }
            for (t, y), r in zip(combos, results)
        ],
    }


@app.get("/api/grid")
def grid():
    a = request.args
    return jsonify(
        _grid(
            float(a.get("score_differential", 0)),
            float(a.get("clock_seconds", 900)),
            int(a.get("qtr", 4)),
            int(a.get("posteam_timeouts", 3)),
            int(a.get("defteam_timeouts", 3)),
            a.get("roof", "outdoors"),
            int(a.get("is_home", 1)),
            float(a.get("spread_line", 0)),
            int(a.get("max_togo", 12)),
            int(a.get("step", 2)),
        )
    )


@app.get("/api/blunders")
def blunders():
    a = request.args
    df = REAL
    if a.get("season"):
        df = df.filter(pl.col("season") == int(a["season"]))
    if a.get("team"):
        df = df.filter(pl.col("posteam") == a["team"])
    if a.get("coach"):
        df = df.filter(pl.col("coach") == a["coach"])
    if a.get("actual"):
        df = df.filter(pl.col("actual") == a["actual"])
    if a.get("recommended"):
        df = df.filter(pl.col("recommended") == a["recommended"])

    limit = min(int(a.get("limit", 50)), 500)
    cols = [
        "season", "week", "posteam", "defteam", "coach", "qtr", "clock",
        "ydstogo", "yardline_100", "field_position", "score_differential",
        "actual", "recommended", "wp_loss", "wp_go", "wp_punt", "wp_field_goal",
        "convert_prob", "fg_make_prob", "desc",
    ]
    rows = df.sort("wp_loss", descending=True).head(limit).select(cols).to_dicts()
    for r in rows:
        for k, v in r.items():
            r[k] = _clean(v, None if k in ("coach", "desc") else 0)
    return jsonify({"total": df.height, "rows": rows})


@app.get("/api/coaches")
def coaches():
    rows = COACHES.sort("mean_wp_loss").to_dicts()
    for r in rows:
        for k, v in r.items():
            r[k] = _clean(v, None if k == "coach" else 0)
    return jsonify({"rows": rows})


@app.get("/api/seasons")
def seasons():
    return jsonify({"rows": SEASONS.sort("season").to_dicts()})


# --------------------------------------------------------------------------
# Serve the built dashboard when it exists, so the whole thing runs from one
# origin in production. In development Vite serves the frontend and proxies
# /api here instead.
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path: str):
    if not DIST_DIR.exists():
        return jsonify({"error": "dashboard not built; run `make dashboard-build`"}), 404
    target = DIST_DIR / path
    if path and target.exists() and target.is_file():
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
