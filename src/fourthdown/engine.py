"""The fourth-down decision engine.

This is where the four models become a recommendation. For each of the three
actions we enumerate what can happen, value the resulting game state, and take
the expectation:

    EV(action) = sum_outcomes  P(outcome) * V(state after outcome)

The recommendation is the argmax. Nothing here is trained -- it is arithmetic
over the model outputs -- and that is the point: there is no ground-truth
"correct decision" label to learn from, so the correctness of the answer rests
entirely on the outcome probabilities and the state values being right.

Two conventions do most of the work, and both are easy to get subtly wrong:

Mirroring. `yardline_100` always counts yards to the end zone the team WITH
the ball is attacking. When possession changes, the same physical spot becomes
`100 - yardline_100`, the score differential negates, timeouts swap, and home
field flips. `GameState.change_possession` does all of it in one place.

Perspective. Win probability is zero-sum: WP(us) = 1 - WP(them). Expected
points are too, but EP has an extra wrinkle -- it measures the NEXT score, so
once a branch scores, that score *is* the answer and we must not add the value
of the ensuing kickoff on top of it. Branches carry `ep_is_terminal` to mark
this. Win probability has no such shortcut and always evaluates the resulting
state, because scoring 7 with two minutes left and scoring 7 in the first
quarter are worth very different amounts of win.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from .binning import GAIN_REP, PUNT_REP, gain_bin_lower_bounds
from .config import (
    FIRST_DYNAMIC_KICKOFF_SEASON,
    MODEL_DIR,
    NEXT_SCORE_VALUES,
    PUNT_RETURN_TD_BIN,
    fg_distance_from_yardline,
)
from .numpy_models import sigmoid, softmax
from .preprocess import Preprocessor

# Seconds from the fourth-down snap to the next scrimmage snap, measured over
# 2016-2025 (see scripts/README notes). These are deliberately taken from the
# same `game_seconds_remaining` field the state model trained on, so the states
# we construct here sit in the same distribution the model learned from.
ELAPSED_GO_CONVERT = 33.0
ELAPSED_GO_FAIL = 19.6
ELAPSED_GO_TD = 8.4
ELAPSED_PUNT = 9.5
ELAPSED_FG_MADE = 6.3
ELAPSED_FG_MISSED = 5.0

# Where a team starts after receiving a kickoff. The 2024 dynamic-kickoff rule
# moved the touchback spot from the 25 to the 30.
def kickoff_start_yardline(season: int) -> float:
    return 70.0 if season >= FIRST_DYNAMIC_KICKOFF_SEASON else 75.0


# A missed field goal gives the defense the ball at the spot of the kick --
# seven yards behind the line of scrimmage -- or its own 20, whichever is
# better for the defense.
FG_SNAP_BACK = 7.0
MISSED_FG_FLOOR_YARDLINE = 80.0

ACTIONS = ("go", "punt", "field_goal")


# ==========================================================================
@dataclass(frozen=True)
class GameState:
    """A pre-snap situation, always from the possessing team's point of view."""

    yardline_100: float
    ydstogo: float
    down: int
    game_seconds_remaining: float
    half_seconds_remaining: float
    score_differential: float
    posteam_timeouts_remaining: int = 3
    defteam_timeouts_remaining: int = 3
    posteam: str = "UNK"
    defteam: str = "UNK"
    is_home: int = 1
    roof: str = "outdoors"
    receive_2h_ko: int = 0
    season: int = 2025
    spread_line: float = 0.0  # posteam perspective; positive means favoured
    wind: float = 0.0
    temp: float = 60.0
    kicker_id: str | None = None

    # ------------------------------------------------------------------
    @property
    def goal_to_go(self) -> int:
        return int(self.ydstogo >= self.yardline_100)

    def _clock(self, elapsed: float) -> dict:
        return {
            "game_seconds_remaining": max(0.0, self.game_seconds_remaining - elapsed),
            "half_seconds_remaining": max(0.0, self.half_seconds_remaining - elapsed),
        }

    def keep_possession(self, new_yardline_100: float, elapsed: float) -> "GameState":
        """Same offense, fresh set of downs."""
        yl = float(np.clip(new_yardline_100, 1.0, 99.0))
        return replace(
            self,
            yardline_100=yl,
            ydstogo=min(10.0, yl),
            down=1,
            **self._clock(elapsed),
        )

    def change_possession(
        self,
        new_yardline_100: float,
        elapsed: float,
        points_for_posteam: float = 0.0,
    ) -> "GameState":
        """Hand the ball over. Returns the state from the NEW offense's view.

        `new_yardline_100` is already expressed in the new offense's frame --
        i.e. its own distance to the end zone it is now attacking.
        """
        yl = float(np.clip(new_yardline_100, 1.0, 99.0))
        return replace(
            self,
            yardline_100=yl,
            ydstogo=min(10.0, yl),
            down=1,
            # Negated because the differential is now measured from the other
            # sideline, after crediting any points the old offense just scored.
            score_differential=-(self.score_differential + points_for_posteam),
            posteam_timeouts_remaining=self.defteam_timeouts_remaining,
            defteam_timeouts_remaining=self.posteam_timeouts_remaining,
            posteam=self.defteam,
            defteam=self.posteam,
            is_home=1 - self.is_home,
            receive_2h_ko=1 - self.receive_2h_ko,
            spread_line=-self.spread_line,
            **self._clock(elapsed),
        )

    def after_scoring_and_kicking_off(
        self, points_for_posteam: float, elapsed: float
    ) -> "GameState":
        """We scored; the opponent receives. Returned from THEIR point of view."""
        return self.change_possession(
            kickoff_start_yardline(self.season), elapsed, points_for_posteam
        )

    def after_conceding_and_receiving(
        self, points_for_opponent: float, elapsed: float
    ) -> "GameState":
        """They scored; we receive. Stays in OUR point of view."""
        return replace(
            self,
            yardline_100=kickoff_start_yardline(self.season),
            ydstogo=10.0,
            down=1,
            score_differential=self.score_differential - points_for_opponent,
            **self._clock(elapsed),
        )


@dataclass
class Branch:
    """One possible outcome of one action."""

    prob: float
    label: str
    points: float           # immediate points, original offense's perspective
    state: GameState | None
    flipped: bool           # `state` belongs to the opponent
    ep_is_terminal: bool    # the next score is settled: it is `points`


# How large a win-probability edge has to be before we call it a decision.
#
# The state model's win-probability calibration error is about 0.027, and the
# conversion model runs ~0.02 hot backed up in a team's own territory. Any
# recommendation resting on a margin thinner than that is noise wearing a
# confident face. Deep-territory fourth-and-medium calls in particular come out
# separated by ~0.003 WP, which is not a real answer, so the engine reports
# them as toss-ups rather than inventing a preference.
TOSSUP_MARGIN = 0.005
LEAN_MARGIN = 0.02


def describe_yardline(yardline_100: float) -> str:
    """Human-readable field position, from the possessing team's perspective."""
    if abs(yardline_100 - 50) < 1e-6:
        return "midfield"
    return f"own {100 - yardline_100:.0f}" if yardline_100 > 50 else f"opp {yardline_100:.0f}"


@dataclass
class DecisionResult:
    ep: dict[str, float]
    wp: dict[str, float]
    recommendation: str
    wp_margin: float          # best minus runner-up, in win probability
    confidence: str           # "clear" | "lean" | "toss-up"
    convert_prob: float
    fg_make_prob: float
    branches: dict[str, list[dict]]

    @property
    def is_tossup(self) -> bool:
        return self.confidence == "toss-up"


# ==========================================================================
def _state_frame(states: list[GameState]) -> pl.DataFrame:
    """Build the exact column set `Preprocessor.transform` expects.

    Includes the era flags that the preprocessor drops. They have to be
    present for the expansion step even though they carry no weight -- the
    fitted column list is what decides which survive.
    """
    elapsed_share = [(3600.0 - s.game_seconds_remaining) / 3600.0 for s in states]
    decay = np.exp(-4.0 * np.array(elapsed_share))
    sd = np.array([s.score_differential for s in states], dtype=np.float64)
    spread = np.array([s.spread_line for s in states], dtype=np.float64)

    return pl.DataFrame(
        {
            "yardline_100": [s.yardline_100 for s in states],
            "ydstogo": [s.ydstogo for s in states],
            "down": [s.down for s in states],
            "goal_to_go": [s.goal_to_go for s in states],
            "half_seconds_remaining": [s.half_seconds_remaining for s in states],
            "game_seconds_remaining": [s.game_seconds_remaining for s in states],
            "posteam_timeouts_remaining": [s.posteam_timeouts_remaining for s in states],
            "defteam_timeouts_remaining": [s.defteam_timeouts_remaining for s in states],
            "is_home": [s.is_home for s in states],
            "roof_dome": [int(s.roof == "dome") for s in states],
            "roof_outdoors": [int(s.roof in ("outdoors", "open")) for s in states],
            "roof_retractable": [int(s.roof in ("closed", "retractable")) for s in states],
            "receive_2h_ko": [s.receive_2h_ko for s in states],
            "season_norm": [(s.season - 2016) / 9.0 for s in states],
            "is_17_game_era": [int(s.season >= 2021) for s in states],
            "dynamic_kickoff_era": [int(s.season >= 2024) for s in states],
            "ot_rules_2025": [int(s.season >= 2025) for s in states],
            "score_differential": sd,
            "diff_time_ratio": sd / decay,
            "spread_time": spread * decay,
            "posteam_spread_line": spread,
            "posteam": [s.posteam for s in states],
            "defteam": [s.defteam for s in states],
        }
    )


class DecisionEngine:
    """Loads every trained model and evaluates fourth-down situations.

    Two inference backends produce identical numbers:

    "numpy" reads `models/weights.npz` and runs the forward passes with numpy
    alone. The four models total 91k parameters -- 342 KB -- so importing 536 MB
    of PyTorch to serve them is the wrong trade for a deployed container. This
    is the default whenever the exported bundle is present.

    "torch" loads the .pt checkpoints. Used in training and by anything that
    needs autograd. `scripts/export_weights.py` asserts the two agree to 1e-4;
    measured worst case is 3.4e-07, which is float32 round-off.
    """

    def __init__(
        self,
        model_dir: Path | str = MODEL_DIR,
        league_average: bool = True,
        backend: str = "auto",
    ):
        model_dir = Path(model_dir)
        self.league_average = league_average
        self.pre = Preprocessor.load(model_dir / "preprocessor")
        self.gain_lower = gain_bin_lower_bounds()
        self.point_values = np.array(NEXT_SCORE_VALUES, dtype=np.float64)

        if backend == "auto":
            backend = "numpy" if (model_dir / "weights.npz").exists() else "torch"
        self.backend = backend

        if backend == "numpy":
            from .numpy_models import NumpyBundle

            self._nb = NumpyBundle(model_dir)
            self.gain_meta = self._nb.meta["gain"]
            self.punt_meta = self._nb.meta["punt"]
            self.fg_meta = self._nb.meta["fg"]
        else:
            self._load_torch(model_dir)

    def _load_torch(self, model_dir: Path) -> None:
        import torch

        from .models import FGNet, GainNet, MultiTaskStateNet, PuntNet

        self._torch = torch
        ck = torch.load(model_dir / "state_net.pt", map_location="cpu", weights_only=False)
        self.state_net = MultiTaskStateNet(ck["n_trunk"], ck["n_wp"], ck["n_teams"])
        self.state_net.load_state_dict(ck["state_dict"]); self.state_net.eval()

        for name, cls, attr in (
            ("fg", FGNet, "fg_net"), ("punt", PuntNet, "punt_net"), ("gain", GainNet, "gain_net")
        ):
            c = torch.load(model_dir / f"{name}_net.pt", map_location="cpu", weights_only=False)
            arg2 = c["n_kickers"] if name == "fg" else c["n_bins"]
            m = cls(c["n_features"], arg2)
            m.load_state_dict(c["state_dict"]); m.eval()
            setattr(self, attr, m)
            setattr(self, f"{name}_meta", c)

    # ------------------------------------------------------------------
    # Backend-neutral forward passes. Everything returns numpy, so the
    # decision arithmetic below never branches on backend.
    def _state_logits(self, d: dict):
        if self.backend == "numpy":
            return self._nb.state(
                d["trunk"].astype(np.float32), d["wp"].astype(np.float32),
                d["posteam"], d["defteam"], league_average=self.league_average,
            )
        t = self._torch
        with t.no_grad():
            o = self.state_net(
                t.tensor(d["trunk"]), t.tensor(d["wp"]),
                t.tensor(d["posteam"]), t.tensor(d["defteam"]),
                league_average=self.league_average,
            )
        return (
            o["next_score_logits"].numpy(),
            o["win_logit"].numpy(),
            o["convert_logit"].numpy(),
        )

    def _mlp_logits(self, which: str, x: np.ndarray) -> np.ndarray:
        meta = getattr(self, f"{which}_meta")
        m = np.array(meta["scaler"]["mean"])
        s = np.array(meta["scaler"]["std"])
        xs = np.clip((x - m) / s, -10, 10)
        if self.backend == "numpy":
            return getattr(self._nb, which)(xs.astype(np.float32))
        t = self._torch
        with t.no_grad():
            return getattr(self, f"{which}_net")(t.tensor(xs.astype(np.float32))).numpy()

    def _fg_logits(self, x: np.ndarray, k: np.ndarray, league_average: bool) -> np.ndarray:
        if self.backend == "numpy":
            return self._nb.fg(x.astype(np.float32), k, league_average=league_average)
        t = self._torch
        with t.no_grad():
            return self.fg_net(
                t.tensor(x.astype(np.float32)), t.tensor(k), league_average=league_average
            ).numpy()

    # ------------------------------------------------------------------
    # A decision grid fans out to ~19,500 branch states in one call. Running
    # those through the trunk in a single batch allocates every hidden
    # activation at once, which peaked around 400 MB resident. Chunking caps
    # the transient allocation without changing any result.
    CHUNK = 4096

    def value_states(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """Expected points and win probability for a batch of states."""
        d = self.pre.transform(_state_frame(states))
        n = d["trunk"].shape[0]
        eps, wps = [], []
        for i in range(0, n, self.CHUNK):
            sl = slice(i, i + self.CHUNK)
            chunk = {k: v[sl] for k, v in d.items()}
            ns_logits, win_logit, _ = self._state_logits(chunk)
            eps.append(softmax(ns_logits, axis=1) @ self.point_values)
            wps.append(sigmoid(win_logit))
        return np.concatenate(eps), np.concatenate(wps)

    # ------------------------------------------------------------------
    # Batched outcome distributions. Everything is vectorized over a list of
    # situations because the decision grid evaluates hundreds at once and the
    # per-call overhead dominates the actual arithmetic.
    def _gain_distributions(self, states: list[GameState]) -> np.ndarray:
        x = np.array(
            [[s.ydstogo, s.yardline_100, 1.0, float(s.goal_to_go)] for s in states]
        )
        return softmax(self._mlp_logits("gain", x), axis=1)

    def _punt_distributions(self, states: list[GameState]) -> np.ndarray:
        x = np.array(
            [
                [s.yardline_100, s.wind, s.temp, float(s.roof in ("dome", "closed"))]
                for s in states
            ]
        )
        return softmax(self._mlp_logits("punt", x), axis=1)

    def _fg_make_probs(self, states: list[GameState]) -> np.ndarray:
        kmap = self.fg_meta["kicker_map"]
        x = np.array(
            [
                [
                    fg_distance_from_yardline(s.yardline_100),
                    s.wind,
                    s.temp,
                    float(s.roof in ("dome", "closed")),
                ]
                for s in states
            ]
        )
        k = np.array([kmap.get(s.kicker_id, 0) for s in states], dtype=np.int64)
        # Mixed known/unknown kickers in one batch: run both ways and select,
        # which is cheaper than splitting the batch.
        specific = sigmoid(self._fg_logits(x, k, False))
        average = sigmoid(self._fg_logits(x, k, True))
        use_avg = np.array(
            [self.league_average or s.kicker_id not in kmap for s in states]
        )
        return np.where(use_avg, average, specific)

    # ------------------------------------------------------------------
    def _go_branches(self, s: GameState, p: np.ndarray) -> list[Branch]:
        branches: list[Branch] = []

        for i, prob in enumerate(p):
            if prob < 1e-6:
                continue
            gain = GAIN_REP[i]

            if gain >= s.yardline_100:
                # Reached the end zone.
                branches.append(
                    Branch(
                        float(prob), f"touchdown ({int(gain)} yd)", 7.0,
                        s.after_scoring_and_kicking_off(7.0, ELAPSED_GO_TD),
                        flipped=True, ep_is_terminal=True,
                    )
                )
            elif gain >= s.ydstogo:
                branches.append(
                    Branch(
                        float(prob), f"convert (+{int(gain)} yd)", 0.0,
                        s.keep_possession(s.yardline_100 - gain, ELAPSED_GO_CONVERT),
                        flipped=False, ep_is_terminal=False,
                    )
                )
            else:
                # Turnover on downs at the spot where the play ended.
                spot = s.yardline_100 - gain
                branches.append(
                    Branch(
                        float(prob), f"fail ({int(gain):+d} yd)", 0.0,
                        s.change_possession(100.0 - spot, ELAPSED_GO_FAIL),
                        flipped=True, ep_is_terminal=False,
                    )
                )
        return branches

    def _punt_branches(self, s: GameState, p: np.ndarray) -> list[Branch]:
        branches: list[Branch] = []

        for i, prob in enumerate(p):
            if prob < 1e-6:
                continue
            if i == PUNT_RETURN_TD_BIN:
                branches.append(
                    Branch(
                        float(prob), "returned for TD", -7.0,
                        s.after_conceding_and_receiving(7.0, ELAPSED_PUNT),
                        flipped=False, ep_is_terminal=True,
                    )
                )
            else:
                # PUNT_REP is already the receiving team's own yardline_100.
                branches.append(
                    Branch(
                        float(prob), f"opp ball at {describe_yardline(PUNT_REP[i])}", 0.0,
                        s.change_possession(PUNT_REP[i], ELAPSED_PUNT),
                        flipped=True, ep_is_terminal=False,
                    )
                )
        return branches

    def _fg_branches(self, s: GameState, p_make: float) -> list[Branch]:
        # On a miss the defense takes the ball at the spot of the kick, or its
        # own 20 if that is better for them.
        miss_spot = min(100.0 - (s.yardline_100 + FG_SNAP_BACK), MISSED_FG_FLOOR_YARDLINE)
        return [
            Branch(
                p_make, "made", 3.0,
                s.after_scoring_and_kicking_off(3.0, ELAPSED_FG_MADE),
                flipped=True, ep_is_terminal=True,
            ),
            Branch(
                1.0 - p_make, "missed", 0.0,
                s.change_possession(miss_spot, ELAPSED_FG_MISSED),
                flipped=True, ep_is_terminal=False,
            ),
        ]

    # ------------------------------------------------------------------
    def evaluate(self, s: GameState) -> DecisionResult:
        return self.evaluate_many([s])[0]

    def evaluate_many(self, states: list[GameState]) -> list[DecisionResult]:
        """Evaluate a batch of situations with one pass per model.

        The decision grid asks for several hundred situations at a time, so
        this collects every branch of every action across the whole batch and
        values them in a single call rather than several hundred.
        """
        if not states:
            return []

        gain_ps = self._gain_distributions(states)
        punt_ps = self._punt_distributions(states)
        fg_ps = self._fg_make_probs(states)

        per_state = [
            {
                "go": self._go_branches(s, gain_ps[i]),
                "punt": self._punt_branches(s, punt_ps[i]),
                "field_goal": self._fg_branches(s, float(fg_ps[i])),
            }
            for i, s in enumerate(states)
        ]

        flat = [b for pa in per_state for a in ACTIONS for b in pa[a]]
        ep_vals, wp_vals = self.value_states([b.state for b in flat])

        results: list[DecisionResult] = []
        k = 0
        for i, s in enumerate(states):
            ep_out: dict[str, float] = {}
            wp_out: dict[str, float] = {}
            detail: dict[str, list[dict]] = {}

            for a in ACTIONS:
                ep_sum = wp_sum = 0.0
                rows = []
                for b in per_state[i][a]:
                    v_ep, v_wp = ep_vals[k], wp_vals[k]
                    k += 1

                    # EP: a scoring branch already IS the next score.
                    ep_branch = (
                        b.points if b.ep_is_terminal else (-v_ep if b.flipped else v_ep)
                    )
                    # WP: always value the resulting state; mirror when the
                    # opponent has the ball.
                    wp_branch = (1.0 - v_wp) if b.flipped else v_wp

                    ep_sum += b.prob * ep_branch
                    wp_sum += b.prob * wp_branch
                    rows.append(
                        {
                            "label": b.label,
                            "prob": round(b.prob, 5),
                            "ep": round(float(ep_branch), 4),
                            "wp": round(float(wp_branch), 5),
                        }
                    )
                ep_out[a] = float(ep_sum)
                wp_out[a] = float(wp_sum)
                detail[a] = sorted(rows, key=lambda r: -r["prob"])[:12]

            ranked = sorted(ACTIONS, key=lambda a: -wp_out[a])
            margin = wp_out[ranked[0]] - wp_out[ranked[1]]

            results.append(
                DecisionResult(
                    ep=ep_out,
                    wp=wp_out,
                    recommendation=ranked[0],
                    wp_margin=margin,
                    confidence=(
                        "toss-up" if margin < TOSSUP_MARGIN
                        else "lean" if margin < LEAN_MARGIN
                        else "clear"
                    ),
                    convert_prob=float(
                        gain_ps[i][self.gain_lower >= s.ydstogo].sum()
                    ),
                    fg_make_prob=float(fg_ps[i]),
                    branches=detail,
                )
            )
        return results
