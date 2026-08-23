"""Train the three action-outcome models.

Run:  python scripts/train_outcomes.py

Each model is checked against a marginal baseline (the unconditional
distribution). A conditional model that cannot beat "ignore the inputs" is not
modelling anything, and for the punt and gain models -- where log loss over 20+
bins has no intuitive scale -- the baseline is the only way to read the number.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fourthdown.binning import gain_to_bin, punt_to_bin
from fourthdown.config import (
    MODEL_DIR,
    N_GAIN_BINS,
    N_PUNT_BINS,
    PROCESSED_DIR,
    REPORT_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    ensure_dirs,
)
from fourthdown.features import MIN_KICKER_ATTEMPTS
from fourthdown.models import FGNet, GainNet, PuntNet

SEED = 1738
BATCH = 4096
PATIENCE = 15


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class Scaler:
    """Standardize on train, freeze, clip at inference."""

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-3)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -10, 10).astype(np.float32)

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def split(df: pl.DataFrame):
    return (
        df.filter(pl.col("season").is_in(TRAIN_SEASONS)),
        df.filter(pl.col("season").is_in(VAL_SEASONS)),
        df.filter(pl.col("season").is_in(TEST_SEASONS)),
    )


def train_classifier(
    model, packs, n_out, dev, lr, epochs, weight_decay, extra_arg=False, param_groups=None
):
    """Shared loop for the categorical/binary outcome models.

    `param_groups` exists so a random-effects model can penalize its random
    effect hard without flattening the fixed effect. Applying one global
    weight decay to the field-goal net shrank the kicker embedding *and* the
    distance curve, leaving the model predicting 64% on 60-64 yard attempts
    that actually convert 41% of the time.
    """
    opt = torch.optim.AdamW(
        param_groups if param_groups is not None else model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        for name in ("train", "val"):
            X, Y, K = packs[name]
            is_train = name == "train"
            model.train(is_train)
            n = X.shape[0]
            order = torch.randperm(n, device=dev) if is_train else torch.arange(n, device=dev)
            tot, nb = 0.0, 0
            for i in range(0, n, BATCH):
                idx = order[i : i + BATCH]
                with torch.set_grad_enabled(is_train):
                    out = model(X[idx], K[idx]) if extra_arg else model(X[idx])
                    loss = (
                        F.binary_cross_entropy_with_logits(out, Y[idx].float())
                        if n_out == 1
                        else F.cross_entropy(out, Y[idx])
                    )
                if is_train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                tot += loss.item()
                nb += 1
            if not is_train:
                v = tot / nb
        sched.step()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best, ep


@torch.no_grad()
def infer(model, X, K=None, extra_arg=False):
    model.eval()
    out = model(X, K) if extra_arg else model(X)
    return out


def marginal_baseline_logloss(train_y: np.ndarray, eval_y: np.ndarray, n_bins: int) -> float:
    p = np.bincount(train_y, minlength=n_bins) / len(train_y)
    return float(-np.mean(np.log(np.clip(p[eval_y], 1e-12, None))))


# ==========================================================================
def main() -> None:
    ensure_dirs()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device()
    print(f"device: {dev}\n")
    report: dict = {}

    # ----------------------------------------------------------------
    # 1. Field goals
    # ----------------------------------------------------------------
    print("=" * 68)
    print("FIELD GOAL MODEL")
    print("=" * 68)
    fg = pl.read_parquet(PROCESSED_DIR / "field_goals.parquet")
    ftr, fva, fte = split(fg)

    # Kicker vocabulary from TRAINING attempts only. Anyone below the volume
    # threshold -- and anyone who debuts after the training window -- maps to
    # index 0, "replacement level", rather than getting a noisy personal
    # estimate from a handful of kicks.
    counts = ftr.group_by("kicker_player_id").len()
    frequent = counts.filter(pl.col("len") >= MIN_KICKER_ATTEMPTS)["kicker_player_id"].to_list()
    kmap = {k: i + 1 for i, k in enumerate([k for k in frequent if k is not None])}
    print(f"kickers with >={MIN_KICKER_ATTEMPTS} training attempts: {len(kmap)} (+1 replacement bucket)")

    FG_COLS = ["fg_distance", "wind_filled", "temp_filled", "is_indoor"]

    def fg_pack(d):
        X = np.column_stack([d[c].cast(pl.Float32).to_numpy() for c in FG_COLS])
        k = np.array([kmap.get(x, 0) for x in d["kicker_player_id"].to_list()], dtype=np.int64)
        return X, d["fg_made"].cast(pl.Int64).to_numpy(), k

    Xtr, ytr, ktr = fg_pack(ftr)
    # The FG model is a GLM whose knots are defined in yards, so it takes raw
    # features. An identity scaler is stored alongside it purely so every
    # saved model loads through the same interface.
    sc = Scaler()
    sc.mean = np.zeros(Xtr.shape[1])
    sc.std = np.ones(Xtr.shape[1])
    packs = {}
    for nm, d in (("train", ftr), ("val", fva), ("test", fte)):
        X, y, k = fg_pack(d)
        packs[nm] = (
            torch.tensor(X.astype(np.float32), device=dev),
            torch.tensor(y, device=dev),
            torch.tensor(k, device=dev),
        )

    fgnet = FGNet(n_features=len(FG_COLS), n_kickers=len(kmap) + 1).to(dev)
    # Split penalties: the kicker random effect is shrunk hard toward league
    # average (otherwise a 25-attempt kicker on a hot streak looks elite),
    # while the shared distance curve is left essentially unpenalized so it can
    # bend as steeply as the data demands at long range.
    fgnet, vloss, eps = train_classifier(
        fgnet, packs, 1, dev, lr=3e-3, epochs=300, weight_decay=0.0, extra_arg=True,
        param_groups=[
            {
                "params": [
                    fgnet.intercept,
                    fgnet.slope_raw,
                    fgnet.wind_x_dist,
                    *fgnet.conditions.parameters(),
                ],
                "weight_decay": 0.0,
            },
            {"params": fgnet.kicker_skill.parameters(), "weight_decay": 5e-2},
        ],
    )
    print(f"trained {eps} epochs, best val loss {vloss:.4f}")

    for nm in ("val", "test"):
        X, y, k = packs[nm]
        p = torch.sigmoid(infer(fgnet, X, k, extra_arg=True)).cpu().numpy()
        yy = y.cpu().numpy()
        base = ytr.mean()
        print(
            f"  {nm.upper():<5} log loss {log_loss(yy, p.clip(1e-6,1-1e-6)):.4f}  "
            f"Brier {brier_score_loss(yy, p):.4f}  AUC {roc_auc_score(yy, p):.4f}  "
            f"(marginal Brier {brier_score_loss(yy, np.full_like(p, base)):.4f})"
        )
        report[f"fg_{nm}_logloss"] = float(log_loss(yy, p.clip(1e-6, 1 - 1e-6)))
        report[f"fg_{nm}_brier"] = float(brier_score_loss(yy, p))

    # Calibration by distance, on real attempts with their real kickers and
    # real conditions. This is the check that matters. Comparing a synthetic
    # calm-weather league-average curve against pooled empirical rates is
    # apples-to-oranges -- the grid holds wind at zero while the actuals
    # include every windy December game in ten seasons.
    print("\n  TEST calibration by distance (real features, real kickers)")
    Xte_f, yte_f, kte_f = fg_pack(fte)
    with torch.no_grad():
        fgnet.eval()
        p_te = torch.sigmoid(
            fgnet(torch.tensor(Xte_f.astype(np.float32), device=dev),
                  torch.tensor(kte_f, device=dev))
        ).cpu().numpy()
    d_te = fte["fg_distance"].to_numpy()
    print(f"    {'distance':<11}{'n':>7}{'pred':>9}{'actual':>9}{'gap':>8}")
    worst = 0.0
    for lo in range(20, 70, 5):
        s = (d_te >= lo) & (d_te < lo + 5)
        if s.sum() >= 15:
            gap = p_te[s].mean() - yte_f[s].mean()
            worst = max(worst, abs(gap))
            print(f"    {lo}-{lo+4:<7}{s.sum():>7,}{p_te[s].mean():>9.3f}"
                  f"{yte_f[s].mean():>9.3f}{gap:>+8.3f}")
    s = d_te >= 55
    if s.sum() >= 15:
        print(f"    {'55+':<11}{s.sum():>7,}{p_te[s].mean():>9.3f}"
              f"{yte_f[s].mean():>9.3f}{p_te[s].mean()-yte_f[s].mean():>+8.3f}")
    report["fg_worst_distance_bin_gap"] = float(worst)

    # Illustrative only: calm, outdoors, league-average kicker. NOT comparable
    # to the pooled empirical rates above.
    print("\n  league-average curve (calm, outdoors, mean kicker)")
    dists = np.arange(20, 66, 5, dtype=np.float32)
    grid = np.column_stack(
        [dists, np.zeros_like(dists), np.full_like(dists, 60.0), np.zeros_like(dists)]
    )
    fgnet.eval()
    with torch.no_grad():
        gp = torch.sigmoid(
            fgnet(torch.tensor(grid.astype(np.float32), device=dev),
                  torch.zeros(len(dists), dtype=torch.long, device=dev),
                  league_average=True)
        ).cpu().numpy()
    print("    " + "  ".join(f"{int(d)}y:{pm:.2f}" for d, pm in zip(dists, gp)))

    # Kicker spread, as a sanity check on the shrinkage. Reported relative to
    # the league mean, since that -- not zero -- is the baseline kicker.
    skill = fgnet.kicker_skill.weight.detach().cpu().numpy().ravel()
    rel = skill - skill.mean()
    base45 = np.log(0.74 / 0.26)
    print(f"\n  kicker skill vs league mean (log-odds): sd={rel.std():.3f}  "
          f"range=[{rel.min():+.2f}, {rel.max():+.2f}]")
    print(f"  -> at a 45-yard kick that spans roughly "
          f"{100/(1+np.exp(-(base45+rel.min()))):.0f}%-"
          f"{100/(1+np.exp(-(base45+rel.max()))):.0f}% make rate")

    # Monotonicity is guaranteed by construction; assert it anyway so a future
    # change to the parameterization cannot quietly break it.
    with torch.no_grad():
        fine = np.arange(18, 76, 1, dtype=np.float32)
        fg_grid = np.column_stack(
            [fine, np.zeros_like(fine), np.full_like(fine, 60.0), np.zeros_like(fine)]
        )
        curve = torch.sigmoid(
            fgnet(torch.tensor(fg_grid, device=dev),
                  torch.zeros(len(fine), dtype=torch.long, device=dev),
                  league_average=True)
        ).cpu().numpy()
    assert np.all(np.diff(curve) <= 1e-9), "FG curve is not monotone decreasing"
    print(f"  monotone over 18-75 yards: OK ({curve[0]:.3f} down to {curve[-1]:.3f})")

    # ----------------------------------------------------------------
    # 2. Punts
    # ----------------------------------------------------------------
    print("\n" + "=" * 68)
    print("PUNT MODEL")
    print("=" * 68)
    pu = pl.read_parquet(PROCESSED_DIR / "punts.parquet")
    ptr, pva, pte = split(pu)
    PUNT_COLS = ["yardline_100", "wind_filled2", "temp_filled2", "is_indoor2"]

    def punt_pack(d):
        d = d.with_columns([
            pl.col("wind").fill_null(0.0).alias("wind_filled2"),
            pl.col("temp").fill_null(60.0).alias("temp_filled2"),
            (pl.col("roof").is_in(["dome", "closed"])).cast(pl.Int8).alias("is_indoor2"),
        ])
        X = np.column_stack([d[c].cast(pl.Float32).to_numpy() for c in PUNT_COLS])
        y = punt_to_bin(
            d["defense_start_yardline"].fill_null(50).cast(pl.Float64).to_numpy(),
            d["return_td"].cast(pl.Int64).to_numpy(),
        )
        return X, y

    Xtr, ytr_p = punt_pack(ptr)
    scp = Scaler().fit(Xtr)
    packs = {}
    for nm, d in (("train", ptr), ("val", pva), ("test", pte)):
        X, y = punt_pack(d)
        packs[nm] = (torch.tensor(scp(X), device=dev), torch.tensor(y, device=dev), None)

    puntnet = PuntNet(n_features=len(PUNT_COLS), n_bins=N_PUNT_BINS).to(dev)
    puntnet, vloss, eps = train_classifier(
        puntnet, packs, N_PUNT_BINS, dev, lr=2e-3, epochs=250, weight_decay=1e-3
    )
    print(f"trained {eps} epochs, best val loss {vloss:.4f}")
    for nm, d in (("val", pva), ("test", pte)):
        X, y, _ = packs[nm]
        p = torch.softmax(infer(puntnet, X), dim=1).cpu().numpy()
        yy = y.cpu().numpy()
        ll = log_loss(yy, p, labels=list(range(N_PUNT_BINS)))
        bl = marginal_baseline_logloss(ytr_p, yy, N_PUNT_BINS)
        print(f"  {nm.upper():<5} log loss {ll:.4f}   marginal baseline {bl:.4f}   "
              f"improvement {bl-ll:+.4f}")
        report[f"punt_{nm}_logloss"] = float(ll)
        report[f"punt_{nm}_baseline"] = float(bl)

    # ----------------------------------------------------------------
    # 3. Go / yards gained
    # ----------------------------------------------------------------
    print("\n" + "=" * 68)
    print("GO (YARDS GAINED) MODEL")
    print("=" * 68)
    cv = pl.read_parquet(PROCESSED_DIR / "conversions.parquet")
    ctr, cva, cte = split(cv)
    GAIN_COLS = ["ydstogo", "yardline_100", "is_fourth", "goal_to_go_f"]

    def gain_pack(d):
        d = d.with_columns([
            (pl.col("down") == 4).cast(pl.Float32).alias("is_fourth"),
            pl.col("goal_to_go").fill_null(0).cast(pl.Float32).alias("goal_to_go_f"),
        ])
        X = np.column_stack([d[c].cast(pl.Float32).to_numpy() for c in GAIN_COLS])
        y = gain_to_bin(d["yards_gained_clipped"].cast(pl.Float64).to_numpy())
        return X, y

    Xtr, ytr_g = gain_pack(ctr)
    scg = Scaler().fit(Xtr)
    packs = {}
    for nm, d in (("train", ctr), ("val", cva), ("test", cte)):
        X, y = gain_pack(d)
        packs[nm] = (torch.tensor(scg(X), device=dev), torch.tensor(y, device=dev), None)

    gainnet = GainNet(n_features=len(GAIN_COLS), n_bins=N_GAIN_BINS).to(dev)
    gainnet, vloss, eps = train_classifier(
        gainnet, packs, N_GAIN_BINS, dev, lr=2e-3, epochs=250, weight_decay=1e-3
    )
    print(f"trained {eps} epochs, best val loss {vloss:.4f}")
    for nm in ("val", "test"):
        X, y, _ = packs[nm]
        p = torch.softmax(infer(gainnet, X), dim=1).cpu().numpy()
        yy = y.cpu().numpy()
        ll = log_loss(yy, p, labels=list(range(N_GAIN_BINS)))
        bl = marginal_baseline_logloss(ytr_g, yy, N_GAIN_BINS)
        print(f"  {nm.upper():<5} log loss {ll:.4f}   marginal baseline {bl:.4f}   "
              f"improvement {bl-ll:+.4f}")
        report[f"gain_{nm}_logloss"] = float(ll)
        report[f"gain_{nm}_baseline"] = float(bl)

    # The distribution model implies a conversion probability. It was never
    # trained on the convert/fail label, so agreement with the empirical rate
    # is an independent check that the bins mean what we think they mean.
    print("\n  implied P(convert) from the gain distribution vs actual, 4th downs, test")
    from fourthdown.binning import gain_bin_lower_bounds

    lower = gain_bin_lower_bounds()
    te4 = cte.filter(pl.col("down") == 4)
    if te4.height:
        X4, _ = gain_pack(te4)
        p4 = torch.softmax(
            infer(gainnet, torch.tensor(scg(X4), device=dev)), dim=1
        ).cpu().numpy()
        togo = te4["ydstogo"].to_numpy()
        implied = np.array([p4[i][lower >= togo[i]].sum() for i in range(len(togo))])
        actual = te4["converted"].to_numpy()
        print(f"    {'distance':<11}{'n':>7}{'implied':>10}{'actual':>9}")
        for dmin, dmax in [(1, 1), (2, 2), (3, 4), (5, 7), (8, 12)]:
            sel = (togo >= dmin) & (togo <= dmax)
            if sel.sum() >= 20:
                lbl = f"{dmin}" if dmin == dmax else f"{dmin}-{dmax}"
                print(f"    4th & {lbl:<5}{sel.sum():>7,}{implied[sel].mean():>10.3f}"
                      f"{actual[sel].mean():>9.3f}")

    # ----------------------------------------------------------------
    torch.save({"state_dict": fgnet.state_dict(), "n_features": len(FG_COLS),
                "n_kickers": len(kmap) + 1, "scaler": sc.as_dict(),
                "kicker_map": kmap, "cols": FG_COLS}, MODEL_DIR / "fg_net.pt")
    torch.save({"state_dict": puntnet.state_dict(), "n_features": len(PUNT_COLS),
                "n_bins": N_PUNT_BINS, "scaler": scp.as_dict(),
                "cols": PUNT_COLS}, MODEL_DIR / "punt_net.pt")
    torch.save({"state_dict": gainnet.state_dict(), "n_features": len(GAIN_COLS),
                "n_bins": N_GAIN_BINS, "scaler": scg.as_dict(),
                "cols": GAIN_COLS}, MODEL_DIR / "gain_net.pt")
    (REPORT_DIR / "outcome_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved outcome models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
