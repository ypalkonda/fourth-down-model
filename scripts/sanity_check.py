"""Sanity-check the decision engine against known football results.

Run:  python scripts/sanity_check.py

A decision engine has no ground-truth label to score against, so correctness
has to be argued rather than measured. These checks are the argument: the
engine has to reproduce results the analytics literature already agrees on
(go on 4th & 1 near midfield, punt on 4th & long from deep in your own end),
and it has to behave monotonically in the ways football requires -- shorter to
go can never be worse, closer to the end zone can never make a field goal less
attractive.

Anything that fails here is a bug in the transition logic, not a modelling
subtlety.
"""

from __future__ import annotations

import numpy as np

from fourthdown.engine import (
    TOSSUP_MARGIN as eng_tossup,
    DecisionEngine,
    GameState,
    describe_yardline,
)

# Tolerance for the monotonicity checks. Set to half a percentage point of win
# probability: large enough to ignore the wobble of a fitted model, small
# enough that a genuine sign error in the transition logic cannot hide under
# it. A tighter bound just flags model noise as a bug.
MONO_TOL = 5e-3

NEUTRAL = dict(
    game_seconds_remaining=1800.0,   # start of the second half
    half_seconds_remaining=1800.0,
    score_differential=0.0,
    posteam_timeouts_remaining=3,
    defteam_timeouts_remaining=3,
    season=2025,
    roof="outdoors",
)

LABEL = {"go": "GO", "punt": "PUNT", "field_goal": "FG"}


def state(yardline_100: float, ydstogo: float, **kw) -> GameState:
    return GameState(
        yardline_100=yardline_100, ydstogo=ydstogo, down=4, **{**NEUTRAL, **kw}
    )


def show(name: str, s: GameState, expect: str | None = None) -> bool:
    r = eng.evaluate(s)
    ok = expect is None or r.recommendation == expect
    mark = "" if expect is None else ("  ok" if ok else f"  <-- EXPECTED {LABEL[expect]}")
    conf = {"clear": "", "lean": " (lean)", "toss-up": " (toss-up)"}[r.confidence]
    print(
        f"  {name:<34} {describe_yardline(s.yardline_100):>9}  4th & {s.ydstogo:<4.0f}"
        f" -> {LABEL[r.recommendation]:<5}{conf:<10}"
        f" | WP go {r.wp['go']:.3f}  punt {r.wp['punt']:.3f}  fg {r.wp['field_goal']:.3f}"
        f" | conv {r.convert_prob:.2f} fg% {r.fg_make_prob:.2f}{mark}"
    )
    return ok


eng = DecisionEngine()
failures = 0

print("=" * 108)
print("CANONICAL SITUATIONS")
print("=" * 108)
for name, s, exp in [
    ("4th & 1 at midfield",            state(50, 1),  "go"),
    ("4th & 1 at opponent 40",         state(40, 1),  "go"),
    ("4th & 1 at own 35",              state(65, 1),  "go"),
    ("4th & 2 at opponent 35",         state(35, 2),  "go"),
    ("4th & 15 at own 20",             state(80, 15), "punt"),
    ("4th & 10 at own 30",             state(70, 10), "punt"),
    ("4th & 8 at own 45",              state(55, 8),  "punt"),
    ("4th & 3 at opponent 5",          state(5, 3),   None),
    ("4th & 10 at opponent 20",        state(20, 10), "field_goal"),
    ("4th & 20 at opponent 25",        state(25, 20), "field_goal"),
    ("4th & goal at 1",                state(1, 1),   None),
]:
    failures += not show(name, s, exp)

print("\n" + "=" * 108)
print("MONOTONICITY (structural requirements -- a failure here is a transition bug)")
print("=" * 108)

# 1. Shorter to go can never lower the value of going for it.
bad = []
worst = 0.0
for yl in (20, 35, 50, 65, 80):
    wps = [eng.evaluate(state(yl, d)).wp["go"] for d in range(1, 11)]
    worst = max(worst, float(np.max(np.diff(wps))))
    if np.any(np.diff(wps) > MONO_TOL):
        bad.append(f"yardline {yl}: {[round(w,3) for w in wps]}")
print(f"  WP(go) decreasing in yards-to-go      : {'OK' if not bad else 'FAIL'}"
      f"   (largest rise {worst:+.4f}, tol {MONO_TOL})")
for b in bad:
    print(f"      {b}")
failures += len(bad)

# 2. Getting closer to the end zone can never lower the value of a field goal.
bad = []
wps = [eng.evaluate(state(yl, 5)).wp["field_goal"] for yl in range(5, 46, 5)]
w2 = float(np.max(np.diff(wps)))
if np.any(np.diff(wps) > MONO_TOL):
    bad.append(f"{[round(w,3) for w in wps]}")
print(f"  WP(fg) decreasing in distance to EZ   : {'OK' if not bad else 'FAIL'}"
      f"   (largest rise {w2:+.4f}, tol {MONO_TOL})")
for b in bad:
    print(f"      {b}")
failures += len(bad)

# 3. Every action's win probability must be a probability.
bad = []
for yl in range(5, 100, 5):
    for d in (1, 5, 10, 20):
        r = eng.evaluate(state(yl, min(d, yl)))
        for a, w in r.wp.items():
            if not (0.0 <= w <= 1.0):
                bad.append(f"yl={yl} togo={d} {a}={w}")
print(f"  all WP within [0, 1]                  : {'OK' if not bad else 'FAIL'}")
failures += len(bad)

print("\n" + "=" * 108)
print("DECISION GRID  (rows: yards to go, cols: yards from opponent end zone)")
print("=" * 108)
cols = list(range(5, 100, 5))
short = {"go": "GO", "punt": "P", "field_goal": "FG"}
print("      " + "".join(f"{c:>5}" for c in cols))
n_toss = 0
for togo in range(1, 11):
    row = ""
    for yl in cols:
        if togo > yl:
            row += "    ."
            continue
        r = eng.evaluate(state(yl, togo))
        cell = short[r.recommendation]
        if r.is_tossup:
            cell = cell.lower() + "?"
            n_toss += 1
        row += f"{cell:>5}"
    print(f"  {togo:>2}  {row}")
print("\n  (GO = go for it, FG = field goal, P = punt, . = impossible)")
print(f"  lowercase + '?' marks a toss-up: the top two options are within "
      f"{eng_tossup:.3f} WP, which is inside the model's own error. "
      f"{n_toss} of the cells shown.")

print("\n" + "=" * 108)
print("LATE-GAME BEHAVIOUR -- where expected points and win probability diverge")
print("=" * 108)
for name, s in [
    ("down 3, 0:40 left, opp 20", state(20, 8, score_differential=-3.0,
                                        game_seconds_remaining=40.0, half_seconds_remaining=40.0)),
    ("down 7, 0:40 left, opp 20", state(20, 8, score_differential=-7.0,
                                        game_seconds_remaining=40.0, half_seconds_remaining=40.0)),
    ("up 2, 2:00 left, own 40",   state(60, 4, score_differential=2.0,
                                        game_seconds_remaining=120.0, half_seconds_remaining=120.0)),
]:
    r = eng.evaluate(s)
    ep_best = max(r.ep, key=r.ep.get)
    print(f"  {name:<28} EP says {LABEL[ep_best]:<5} | WP says {LABEL[r.recommendation]:<5}"
          f" | EP go {r.ep['go']:+.2f} punt {r.ep['punt']:+.2f} fg {r.ep['field_goal']:+.2f}"
          f" | WP go {r.wp['go']:.3f} punt {r.wp['punt']:.3f} fg {r.wp['field_goal']:.3f}")

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
