"""Export trained weights to a torch-free bundle, and prove it matches.

Run:  python scripts/export_weights.py

Writes models/weights.npz and models/weights_meta.json, then re-runs both
implementations on random inputs and reports the largest disagreement. If the
numpy path ever drifts from the PyTorch path, this fails loudly here rather
than silently serving different numbers in production.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from fourthdown.config import MODEL_DIR
from fourthdown.models import FGNet, GainNet, MultiTaskStateNet, PuntNet
from fourthdown.numpy_models import NumpyBundle, sigmoid, softmax

TOL = 1e-4


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}

    state_ck = torch.load(MODEL_DIR / "state_net.pt", map_location="cpu", weights_only=False)
    for k, v in state_ck["state_dict"].items():
        arrays[f"state/{k}"] = v.numpy()
    meta["state"] = {k: state_ck[k] for k in ("n_trunk", "n_wp", "n_teams")}

    for name, cls in (("gain", GainNet), ("punt", PuntNet), ("fg", FGNet)):
        ck = torch.load(MODEL_DIR / f"{name}_net.pt", map_location="cpu", weights_only=False)
        for k, v in ck["state_dict"].items():
            arrays[f"{name}/{k}"] = v.numpy()
        meta[name] = {
            "n_features": ck["n_features"],
            "cols": ck["cols"],
            "scaler": ck["scaler"],
        }
        if name == "fg":
            meta[name]["kicker_map"] = ck["kicker_map"]
            meta[name]["n_kickers"] = ck["n_kickers"]
        else:
            meta[name]["n_bins"] = ck["n_bins"]

    np.savez_compressed(MODEL_DIR / "weights.npz", **arrays)
    (MODEL_DIR / "weights_meta.json").write_text(json.dumps(meta, indent=2))

    total = sum(v.size for v in arrays.values())
    size_kb = (MODEL_DIR / "weights.npz").stat().st_size / 1024
    print(f"exported {total:,} parameters -> weights.npz ({size_kb:.0f} KB)")

    # ------------------------------------------------------------------
    # Equivalence check against the PyTorch implementations.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    nb = NumpyBundle(MODEL_DIR)
    worst: dict[str, float] = {}

    net = MultiTaskStateNet(meta["state"]["n_trunk"], meta["state"]["n_wp"], meta["state"]["n_teams"])
    net.load_state_dict(state_ck["state_dict"]); net.eval()
    N = 256
    tr = rng.normal(size=(N, meta["state"]["n_trunk"])).astype(np.float32)
    wp = rng.normal(size=(N, meta["state"]["n_wp"])).astype(np.float32)
    pos = rng.integers(0, meta["state"]["n_teams"], N)
    dfn = rng.integers(0, meta["state"]["n_teams"], N)

    for la in (False, True):
        with torch.no_grad():
            o = net(torch.tensor(tr), torch.tensor(wp), torch.tensor(pos), torch.tensor(dfn),
                    league_average=la)
        ns_t = torch.softmax(o["next_score_logits"], 1).numpy()
        wp_t = torch.sigmoid(o["win_logit"]).numpy()
        cv_t = torch.sigmoid(o["convert_logit"]).numpy()

        ns_l, wp_l, cv_l = nb.state(tr.astype(np.float64), wp.astype(np.float64), pos, dfn, la)
        tag = "league_avg" if la else "team"
        worst[f"state.next_score[{tag}]"] = float(np.abs(softmax(ns_l, 1) - ns_t).max())
        worst[f"state.win[{tag}]"] = float(np.abs(sigmoid(wp_l) - wp_t).max())
        worst[f"state.convert[{tag}]"] = float(np.abs(sigmoid(cv_l) - cv_t).max())

    for name, cls, np_model in (("gain", GainNet, nb.gain), ("punt", PuntNet, nb.punt)):
        m = cls(meta[name]["n_features"], meta[name]["n_bins"])
        m.load_state_dict({k[len(name) + 1 :]: torch.tensor(v)
                           for k, v in arrays.items() if k.startswith(name + "/")})
        m.eval()
        x = rng.normal(size=(N, meta[name]["n_features"])).astype(np.float32)
        with torch.no_grad():
            t = torch.softmax(m(torch.tensor(x)), 1).numpy()
        worst[name] = float(np.abs(softmax(np_model(x.astype(np.float64)), 1) - t).max())

    fg = FGNet(meta["fg"]["n_features"], meta["fg"]["n_kickers"])
    fg.load_state_dict({k[3:]: torch.tensor(v) for k, v in arrays.items() if k.startswith("fg/")})
    fg.eval()
    xf = np.column_stack([
        rng.uniform(18, 70, N), rng.uniform(0, 25, N),
        rng.uniform(10, 95, N), rng.integers(0, 2, N),
    ]).astype(np.float32)
    kf = rng.integers(0, meta["fg"]["n_kickers"], N)
    for la in (False, True):
        with torch.no_grad():
            t = torch.sigmoid(fg(torch.tensor(xf), torch.tensor(kf), league_average=la)).numpy()
        n_ = sigmoid(nb.fg(xf.astype(np.float64), kf, league_average=la))
        worst[f"fg[{'league_avg' if la else 'kicker'}]"] = float(np.abs(n_ - t).max())

    print("\nnumpy vs torch, max absolute difference in probability:")
    for k, v in worst.items():
        print(f"  {k:<28} {v:.2e}  {'ok' if v < TOL else 'FAIL'}")

    bad = {k: v for k, v in worst.items() if v >= TOL}
    if bad:
        raise SystemExit(f"\nnumpy path diverges from torch: {bad}")
    print(f"\nall within {TOL:g} -- numpy inference is equivalent")


if __name__ == "__main__":
    main()
