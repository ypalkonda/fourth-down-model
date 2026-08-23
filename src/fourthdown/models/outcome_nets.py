"""Outcome models for the three fourth-down actions.

Each one answers "if the coach does X, what happens?", and each returns a
distribution rather than an average, because the decision engine has to value
every branch separately.

    FGNet    P(make | distance, conditions, kicker)
    PuntNet  P(receiving team starts at each field position | punt LOS)
    GainNet  P(yards gained falls in each bin | down, distance, field position)

These are separate models rather than more heads on the state network. Their
training sets are different subsets of different sizes with different feature
spaces -- a field-goal model wants kick distance and wind, a gain model wants
down and distance -- and forcing them to share a trunk would be multi-tasking
for its own sake rather than because the tasks share structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(n_in: int, hidden: tuple[int, ...], n_out: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [n_in, *hidden]
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], n_out))
    return nn.Sequential(*layers)


class FGNet(nn.Module):
    """Field-goal make probability: a logistic GLM with a kicker random effect.

    This one is deliberately NOT a neural net, and the reason is worth stating
    because it cuts against the rest of the project.

    An MLP version of this model was well calibrated out to 54 yards and then
    fell apart: it predicted 60% on 60-64 yard attempts that convert 41% of the
    time. Only 98 of 10,731 attempts in ten seasons are longer than 60 yards,
    so binary cross-entropy barely registers them, and an unconstrained network
    flattens out in the tail rather than continuing to fall. That error lands
    exactly where fourth-down decisions are hardest -- "kick the 57-yarder or
    punt?" -- so it is the one place the model cannot afford to be wrong.

    Empirically the log-odds of making a kick are close to linear in distance,
    at roughly -0.11 per yard across the whole range. Encoding that directly
    buys two things a network cannot offer here: the curve is monotone
    decreasing by construction (the distance coefficient is forced negative
    via softplus), and it extrapolates sensibly into the sparse tail instead of
    inventing a plateau.

    The kicker term is a single scalar per kicker added to the logit -- a
    random-effects model, with the effect shrunk toward league average by
    weight decay applied to this embedding only. One dimension is the right
    inductive bias: kickers differ mainly in overall leg, and ~10k attempts
    over ~150 kickers cannot support anything richer.

    Feature order is fixed: [distance, wind, temp, is_indoor].
    """

    # Knots in yards. A single straight line in log-odds could not hold both
    # ends of the curve at once -- fitting the 60+ tail dragged 20-24 yard
    # chip shots down to 90% when they convert at 99%. These let the curve
    # steepen as distance grows while every added slope term is forced
    # negative, so monotonicity survives.
    KNOTS = (28.0, 36.0, 44.0, 52.0, 58.0)

    def __init__(self, n_features: int, n_kickers: int, **_: object) -> None:
        super().__init__()
        self.intercept = nn.Parameter(torch.tensor(5.0))
        # Base slope and one additional slope per knot, each entering as
        # -softplus(.), so the log-odds are monotone decreasing in distance
        # regardless of what the optimizer does, and extrapolate linearly
        # past the final knot rather than inventing a plateau.
        self.slope_raw = nn.Parameter(torch.full((1 + len(self.KNOTS),), -2.0))
        # Wind, temperature, indoor.
        self.conditions = nn.Linear(n_features - 1, 1, bias=False)
        nn.init.zeros_(self.conditions.weight)
        # Wind hurts a 55-yarder more than a chip shot.
        self.wind_x_dist = nn.Parameter(torch.zeros(1))

        self.kicker_skill = nn.Embedding(n_kickers, 1)
        nn.init.zeros_(self.kicker_skill.weight)

    def forward(
        self, x: torch.Tensor, kicker: torch.Tensor, league_average: bool = False
    ) -> torch.Tensor:
        # Features arrive UNSCALED here: knot positions are in yards, and the
        # /10 below is only for optimizer conditioning.
        dist, wind, rest = x[:, 0], x[:, 1], x[:, 1:]
        sp = nn.functional.softplus(self.slope_raw)

        decline = sp[0] * (dist / 10.0)
        for j, c in enumerate(self.KNOTS):
            decline = decline + sp[j + 1] * torch.clamp(dist - c, min=0.0) / 10.0

        logit = (
            self.intercept
            - decline
            + self.conditions(rest).squeeze(-1)
            + self.wind_x_dist * wind * (dist / 10.0)
        )

        if league_average:
            # The MEAN kicker, not the zero vector. The embedding has no
            # reason to centre itself at zero -- fitted skills here run
            # entirely positive -- so zeroing it silently shifts the whole
            # curve down by the average kicker effect.
            logit = logit + self.kicker_skill.weight.mean()
        else:
            logit = logit + self.kicker_skill(kicker).squeeze(-1)
        return logit


class PuntNet(nn.Module):
    """Categorical distribution over where the receiving team takes over.

    Output index `PUNT_RETURN_TD_BIN` is the absorbing "returned for a
    touchdown" outcome, which has no field position to report and has to be
    valued as a score rather than as a spot.
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = _mlp(n_features, hidden, n_bins, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GainNet(nn.Module):
    """Categorical distribution over binned yards gained on a go attempt."""

    def __init__(
        self,
        n_features: int,
        n_bins: int,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = _mlp(n_features, hidden, n_bins, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
