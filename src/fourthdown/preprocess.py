"""Feature scaling and team vocabulary.

Fit statistics come from the TRAINING SEASONS ONLY and are then frozen. If we
standardized using the full dataset, information from 2024-25 would leak into
the 2016-23 training features and the held-out evaluation would flatter us.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .config import NEXT_SCORE_INDEX, TRUNK_FEATURES, WP_HEAD_FEATURES

# `down` is ordinal but the jumps are not uniform in value -- 3rd down is much
# closer to 4th than 1st is to 2nd -- so it gets one-hot encoded rather than
# scaled as a number.
DOWN_VALUES = [1, 2, 3, 4]


# A feature that never varies in the training split carries no learnable
# signal, and standardizing it is actively dangerous: dividing by a ~zero std
# maps any nonzero value at inference to something astronomically large.
#
# This bites here for a concrete reason. `dynamic_kickoff_era` is 0 for every
# play in 2016-2023 and 1 from 2024 on, so with a temporal split it is constant
# in training and set in validation. The first version of this model divided
# it by a clamped std of 1e-6 and fed 1,000,000 into the first linear layer for
# every held-out row. EP bias grew from +0.47 in 2024 to +1.44 in 2025 as more
# such flags switched on.
#
# The real lesson is not "clamp harder" -- it is that you cannot learn the
# effect of a rule change that never occurs in your training data. Those
# features get dropped, loudly, and the README states the limitation.
ZERO_VARIANCE_TOL = 1e-8

# Guard against extrapolation more generally: nothing real is 10 standard
# deviations from the training mean, so anything that far out is a scaling
# artifact rather than a football situation.
CLIP_SIGMA = 10.0


@dataclass
class Preprocessor:
    trunk_cols: list[str] = field(default_factory=list)
    wp_cols: list[str] = field(default_factory=list)
    trunk_mean: np.ndarray | None = None
    trunk_std: np.ndarray | None = None
    wp_mean: np.ndarray | None = None
    wp_std: np.ndarray | None = None
    teams: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    # Raw-space bounds for monotone-in-time features, so that predicting a
    # future season clamps to the most recent one seen rather than
    # extrapolating off the end of the fitted range.
    season_norm_range: tuple[float, float] = (0.0, 1.0)

    # ---------------------------------------------------------------
    @property
    def team_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    @property
    def n_trunk(self) -> int:
        return len(self.trunk_cols)

    @property
    def n_wp(self) -> int:
        return len(self.wp_cols)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    # ---------------------------------------------------------------
    @staticmethod
    def _expand(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
        """Materialize a feature block, one-hotting `down` if present."""
        out_cols: list[str] = []
        blocks: list[np.ndarray] = []
        for c in cols:
            if c == "down":
                for d in DOWN_VALUES:
                    blocks.append((df["down"] == d).cast(pl.Float32).to_numpy())
                    out_cols.append(f"down_{d}")
            else:
                blocks.append(df[c].cast(pl.Float32).to_numpy())
                out_cols.append(c)
        return np.column_stack(blocks).astype(np.float32), out_cols

    @classmethod
    def _expand_selected(
        cls, df: pl.DataFrame, source_cols: list[str], keep_cols: list[str]
    ) -> np.ndarray:
        """Expand the raw feature list, then take only the retained columns
        in the exact order recorded at fit time."""
        block, cols = cls._expand(df, source_cols)
        pos = {c: i for i, c in enumerate(cols)}
        return block[:, [pos[c] for c in keep_cols]]

    # ---------------------------------------------------------------
    def fit(self, train_df: pl.DataFrame) -> "Preprocessor":
        trunk, trunk_cols = self._expand(train_df, TRUNK_FEATURES)
        wp, wp_cols = self._expand(train_df, WP_HEAD_FEATURES)

        # Drop anything constant in training, in both blocks.
        self.dropped = []
        trunk, self.trunk_cols = self._drop_constant(trunk, trunk_cols)
        wp, self.wp_cols = self._drop_constant(wp, wp_cols)

        self.trunk_mean = trunk.mean(axis=0)
        self.trunk_std = np.maximum(trunk.std(axis=0), 1e-3)
        self.wp_mean = wp.mean(axis=0)
        self.wp_std = np.maximum(wp.std(axis=0), 1e-3)

        sn = train_df["season_norm"].cast(pl.Float64)
        self.season_norm_range = (float(sn.min()), float(sn.max()))

        teams = set(train_df["posteam"].drop_nulls().unique().to_list())
        teams |= set(train_df["defteam"].drop_nulls().unique().to_list())
        self.teams = sorted(teams)
        return self

    def _drop_constant(
        self, block: np.ndarray, cols: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        std = block.std(axis=0)
        keep = std > ZERO_VARIANCE_TOL
        self.dropped += [c for c, k in zip(cols, keep) if not k]
        return block[:, keep], [c for c, k in zip(cols, keep) if k]

    # ---------------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        # Clamp the season index into the fitted range. A 2025 play is scored
        # as though it were the most recent season the model actually saw,
        # which is honest extrapolation behaviour rather than running the
        # linear term off the end of its support.
        lo, hi = self.season_norm_range
        df = df.with_columns(pl.col("season_norm").clip(lo, hi))

        trunk = self._expand_selected(df, TRUNK_FEATURES, self.trunk_cols)
        wp = self._expand_selected(df, WP_HEAD_FEATURES, self.wp_cols)

        trunk = np.clip(
            (trunk - self.trunk_mean) / self.trunk_std, -CLIP_SIGMA, CLIP_SIGMA
        )
        wp = np.clip((wp - self.wp_mean) / self.wp_std, -CLIP_SIGMA, CLIP_SIGMA)

        idx = self.team_index
        # Teams absent from the training vocabulary (relocations, e.g. OAK->LV,
        # SD->LAC, STL->LA) fall back to index 0; the model can also be run in
        # league-average mode, which ignores the embedding entirely.
        pos = np.array([idx.get(t, 0) for t in df["posteam"].to_list()], dtype=np.int64)
        dfn = np.array([idx.get(t, 0) for t in df["defteam"].to_list()], dtype=np.int64)

        out = {"trunk": trunk, "wp": wp, "posteam": pos, "defteam": dfn}

        if "next_score_type" in df.columns:
            out["next_score"] = np.array(
                [NEXT_SCORE_INDEX[s] for s in df["next_score_type"].to_list()],
                dtype=np.int64,
            )
        if "posteam_won" in df.columns:
            out["won"] = df["posteam_won"].cast(pl.Float32).to_numpy()
        if "converted" in df.columns:
            out["converted"] = (
                df["converted"].fill_null(0).cast(pl.Float32).to_numpy()
            )
        if "convert_mask" in df.columns:
            out["convert_mask"] = df["convert_mask"].cast(pl.Float32).to_numpy()
        if "ep_sample_weight" in df.columns:
            out["ep_weight"] = df["ep_sample_weight"].cast(pl.Float32).to_numpy()

        return out

    # ---------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path.with_suffix(".npz"),
            trunk_mean=self.trunk_mean,
            trunk_std=self.trunk_std,
            wp_mean=self.wp_mean,
            wp_std=self.wp_std,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "trunk_cols": self.trunk_cols,
                    "wp_cols": self.wp_cols,
                    "teams": self.teams,
                    "dropped": self.dropped,
                    "season_norm_range": list(self.season_norm_range),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        path = Path(path)
        arr = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            trunk_cols=meta["trunk_cols"],
            wp_cols=meta["wp_cols"],
            teams=meta["teams"],
            dropped=meta.get("dropped", []),
            season_norm_range=tuple(meta.get("season_norm_range", (0.0, 1.0))),
            trunk_mean=arr["trunk_mean"],
            trunk_std=arr["trunk_std"],
            wp_mean=arr["wp_mean"],
            wp_std=arr["wp_std"],
        )
