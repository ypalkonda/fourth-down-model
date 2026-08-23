"""Mapping continuous outcomes onto the categorical bins the models predict.

Kept in one place so training and inference cannot drift apart -- a bin index
that means "gained 3 yards" at fit time and "gained 4 yards" at serve time is
the kind of bug that produces plausible, confidently wrong recommendations.
"""

from __future__ import annotations

import numpy as np

from .config import (
    GAIN_BIN_LOWER,
    GAIN_BIN_UPPER,
    GAIN_REPRESENTATIVE,
    N_GAIN_BINS,
    N_PUNT_FIELD_BINS,
    PUNT_BIN_WIDTH,
    PUNT_RETURN_TD_BIN,
)

_GAIN_UPPER = np.array(GAIN_BIN_UPPER, dtype=np.float64)
GAIN_REP = np.array(GAIN_REPRESENTATIVE, dtype=np.float64)

# Midpoint field position for each punt bin, in the receiving team's frame.
PUNT_REP = np.array(
    [i * PUNT_BIN_WIDTH + PUNT_BIN_WIDTH / 2 for i in range(N_PUNT_FIELD_BINS)],
    dtype=np.float64,
)


def gain_to_bin(yards: np.ndarray) -> np.ndarray:
    """Index of the yards-gained bin containing each value."""
    idx = np.searchsorted(_GAIN_UPPER, np.asarray(yards, dtype=np.float64), side="left")
    return np.clip(idx, 0, N_GAIN_BINS - 1).astype(np.int64)


def punt_to_bin(defense_yardline: np.ndarray, return_td: np.ndarray) -> np.ndarray:
    """Index of the punt-outcome bin.

    A return touchdown gets the absorbing bin regardless of field position,
    since there is no "spot" to value -- the points already happened.
    """
    yl = np.asarray(defense_yardline, dtype=np.float64)
    b = np.clip((yl // PUNT_BIN_WIDTH).astype(np.int64), 0, N_PUNT_FIELD_BINS - 1)
    return np.where(np.asarray(return_td) == 1, PUNT_RETURN_TD_BIN, b)


def gain_bin_lower_bounds() -> np.ndarray:
    """Lower edge of each gain bin, used to test feasibility of a conversion."""
    return np.array(GAIN_BIN_LOWER, dtype=np.float64)
