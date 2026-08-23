"""Multi-task game-state value network.

One shared trunk, three heads:

    next_score  7-class softmax -> expected points (dot with point values)
    win         binary          -> win probability
    convert     binary          -> P(convert) on a 3rd/4th down attempt

Why multi-task rather than three separate models -- this is the honest
justification for reaching for a neural net on tabular data at all. All three
targets are downstream of the same latent quantity: how good is this spot on
the field, at this down and distance, with this much time left. Sharing a
trunk lets the conversion head (which sees only ~76k rows) borrow the
representation learned from all ~384k plays, and regularizes each task against
the others. Three independent XGBoost models cannot share anything; the
baseline in `baseline.py` measures whether that sharing actually buys us
anything, and the README reports the answer either way.

The convert head's loss is masked: punts and first-down plays have no
conversion label, so they contribute to the trunk through the other two tasks
only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskStateNet(nn.Module):
    def __init__(
        self,
        n_trunk_features: int,
        n_wp_features: int,
        n_teams: int,
        team_emb_dim: int = 8,
        trunk_hidden: tuple[int, ...] = (256, 128, 64),
        head_hidden: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_teams = n_teams
        self.team_emb_dim = team_emb_dim

        # Separate embeddings: a team's offense and its defense are different
        # things, and the same club can be good at one and bad at the other.
        self.posteam_emb = nn.Embedding(n_teams, team_emb_dim)
        self.defteam_emb = nn.Embedding(n_teams, team_emb_dim)
        nn.init.normal_(self.posteam_emb.weight, std=0.01)
        nn.init.normal_(self.defteam_emb.weight, std=0.01)

        dims = [n_trunk_features + 2 * team_emb_dim, *trunk_hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        d = trunk_hidden[-1]
        self.next_score_head = nn.Sequential(
            nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 7)
        )
        # Score differential and its time interactions enter here, not in the
        # trunk -- see config.TRUNK_FEATURES for the reasoning.
        self.win_head = nn.Sequential(
            nn.Linear(d + n_wp_features, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )
        self.convert_head = nn.Sequential(
            nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1)
        )

    # ------------------------------------------------------------------
    def _team_vectors(
        self, posteam: torch.Tensor, defteam: torch.Tensor, league_average: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up team embeddings, or substitute the league mean.

        League-average mode gives us a generic, team-agnostic recommendation
        from the same trained weights -- useful for the dashboard's "any team"
        scenario input, and for not baking a specific roster into a general
        claim about fourth-down strategy.
        """
        if league_average:
            n = posteam.shape[0]
            p = self.posteam_emb.weight.mean(dim=0, keepdim=True).expand(n, -1)
            d = self.defteam_emb.weight.mean(dim=0, keepdim=True).expand(n, -1)
            return p, d
        return self.posteam_emb(posteam), self.defteam_emb(defteam)

    def forward(
        self,
        trunk_x: torch.Tensor,
        wp_x: torch.Tensor,
        posteam: torch.Tensor,
        defteam: torch.Tensor,
        league_average: bool = False,
    ) -> dict[str, torch.Tensor]:
        p_emb, d_emb = self._team_vectors(posteam, defteam, league_average)
        h = self.trunk(torch.cat([trunk_x, p_emb, d_emb], dim=1))

        return {
            "next_score_logits": self.next_score_head(h),
            "win_logit": self.win_head(torch.cat([h, wp_x], dim=1)).squeeze(-1),
            "convert_logit": self.convert_head(h).squeeze(-1),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def expected_points(
        next_score_logits: torch.Tensor, point_values: torch.Tensor
    ) -> torch.Tensor:
        """EP = sum_k P(next score is class k) * points(k)."""
        return torch.softmax(next_score_logits, dim=1) @ point_values
