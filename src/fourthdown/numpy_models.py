"""Dependency-free inference for the trained models.

The four models together are 91,083 parameters -- 356 KB as float32. Serving
them through PyTorch means shipping 536 MB of framework to run 356 KB of
weights, which is the difference between a container that fits a free tier and
one that does not.

Every layer used at inference is a matmul, a LayerNorm, a SiLU, or an embedding
lookup. All four are a few lines of numpy, and dropout is the identity in eval
mode, so nothing is lost. `scripts/export_weights.py` writes the weights to a
single .npz; this module runs them.

Verified against the PyTorch implementation to ~1e-6 (see the export script's
self-check), which is float32 round-off, not a reimplementation difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Branch on sign so the exponential never overflows for large |x|.
    out = np.empty(x.shape, dtype=x.dtype if x.dtype.kind == "f" else np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class Sequential:
    """Replays an nn.Sequential of Linear / LayerNorm / SiLU from its weights.

    Torch records only parameterized modules in a state_dict, so activations
    have to be reconstructed. The pattern is unambiguous: a Linear is followed
    by a LayerNorm when one is present at the next index, and a SiLU follows
    every block except the final one -- except in the shared trunk, which ends
    on an activation. `final_activation` carries that one difference rather
    than trying to infer it.
    """

    def __init__(self, params: dict, prefix: str, final_activation: bool):
        keys = [k for k in params if k.startswith(prefix + ".")]
        idx = sorted({int(k[len(prefix) + 1 :].split(".")[0]) for k in keys})
        self.blocks = []
        i = 0
        while i < len(idx):
            w = params[f"{prefix}.{idx[i]}.weight"]
            b = params[f"{prefix}.{idx[i]}.bias"]
            ln = None
            if i + 1 < len(idx):
                nxt = params[f"{prefix}.{idx[i + 1]}.weight"]
                if nxt.ndim == 1:  # LayerNorm
                    ln = (nxt, params[f"{prefix}.{idx[i + 1]}.bias"])
                    i += 1
            self.blocks.append((w, b, ln))
            i += 1
        self.final_activation = final_activation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        last = len(self.blocks) - 1
        for j, (w, b, ln) in enumerate(self.blocks):
            x = x @ w.T + b
            if ln is not None:
                x = layer_norm(x, ln[0], ln[1])
            if j < last or self.final_activation:
                x = silu(x)
        return x


class NumpyStateNet:
    """Multi-task state-value network: next score, win, conversion."""

    def __init__(self, params: dict):
        self.pos_emb = params["posteam_emb.weight"]
        self.def_emb = params["defteam_emb.weight"]
        self.trunk = Sequential(params, "trunk", final_activation=True)
        self.next_score = Sequential(params, "next_score_head", final_activation=False)
        self.win = Sequential(params, "win_head", final_activation=False)
        self.convert = Sequential(params, "convert_head", final_activation=False)

    def __call__(self, trunk_x, wp_x, posteam, defteam, league_average=False):
        n = trunk_x.shape[0]
        if league_average:
            p = np.repeat(self.pos_emb.mean(axis=0, keepdims=True), n, axis=0)
            d = np.repeat(self.def_emb.mean(axis=0, keepdims=True), n, axis=0)
        else:
            p, d = self.pos_emb[posteam], self.def_emb[defteam]
        h = self.trunk(np.concatenate([trunk_x, p, d], axis=1))
        return (
            self.next_score(h),
            self.win(np.concatenate([h, wp_x], axis=1))[:, 0],
            self.convert(h)[:, 0],
        )


class NumpyMLPNet:
    """PuntNet / GainNet: a plain MLP producing categorical logits."""

    def __init__(self, params: dict):
        self.net = Sequential(params, "net", final_activation=False)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.net(x)


class NumpyFGNet:
    """Field-goal GLM: monotone piecewise-linear in distance plus a kicker effect."""

    KNOTS = (28.0, 36.0, 44.0, 52.0, 58.0)

    def __init__(self, params: dict):
        # `intercept` is a 0-d tensor and `wind_x_dist` a 1-element vector;
        # flatten both to plain floats so indexing does not depend on which.
        self.intercept = float(np.asarray(params["intercept"]).ravel()[0])
        self.slope_raw = np.asarray(params["slope_raw"]).ravel()
        self.conditions = params["conditions.weight"]
        self.wind_x_dist = float(np.asarray(params["wind_x_dist"]).ravel()[0])
        self.kicker = params["kicker_skill.weight"]

    def __call__(self, x: np.ndarray, kicker: np.ndarray, league_average=False):
        dist, wind, rest = x[:, 0], x[:, 1], x[:, 1:]
        sp = softplus(self.slope_raw)
        decline = sp[0] * (dist / 10.0)
        for j, c in enumerate(self.KNOTS):
            decline = decline + sp[j + 1] * np.clip(dist - c, 0.0, None) / 10.0
        logit = (
            self.intercept
            - decline
            + (rest @ self.conditions.T)[:, 0]
            + self.wind_x_dist * wind * (dist / 10.0)
        )
        skill = (
            self.kicker.mean() if league_average else self.kicker[kicker][:, 0]
        )
        return logit + skill


class NumpyBundle:
    """All four models plus the scalers, loaded from one .npz + one .json."""

    def __init__(self, model_dir: Path | str):
        model_dir = Path(model_dir)
        z = np.load(model_dir / "weights.npz")
        self.meta = json.loads((model_dir / "weights_meta.json").read_text())

        # float32, matching the precision the models were trained and
        # exported in. float64 doubles every intermediate activation for no
        # accuracy -- on a batched grid call that alone cost ~200 MB of peak
        # resident memory.
        def group(prefix: str) -> dict:
            return {
                k[len(prefix) + 1 :]: z[k].astype(np.float32)
                for k in z.files
                if k.startswith(prefix + "/")
            }

        self.state = NumpyStateNet(group("state"))
        self.gain = NumpyMLPNet(group("gain"))
        self.punt = NumpyMLPNet(group("punt"))
        self.fg = NumpyFGNet(group("fg"))
