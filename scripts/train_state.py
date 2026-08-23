"""Train the multi-task state-value network.

Run:  python scripts/train_state.py

Reports next-score log loss, EP agreement with nflfastR, win-probability
calibration, and conversion-model quality on held-out seasons.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fourthdown.config import (
    MODEL_DIR,
    NEXT_SCORE_CLASSES,
    NEXT_SCORE_VALUES,
    PROCESSED_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    ensure_dirs,
)
from fourthdown.models import MultiTaskStateNet
from fourthdown.preprocess import Preprocessor

SEED = 1738
EPOCHS = 60
BATCH = 8192
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

# Relative task weights. Next-score and win are the two we actually serve;
# conversion is a genuine third task but gets a smaller weight because it sees
# far fewer labelled rows and we do not want it dominating the trunk.
W_NEXT_SCORE = 1.0
W_WIN = 1.0
W_CONVERT = 0.5


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensors(d: dict[str, np.ndarray], dev: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        t = torch.as_tensor(v)
        out[k] = t.to(dev)
    return out


def run_epoch(model, data, opt, point_values, train: bool):
    n = data["trunk"].shape[0]
    model.train(train)
    perm = torch.randperm(n, device=data["trunk"].device) if train else torch.arange(n, device=data["trunk"].device)

    totals = np.zeros(5)  # loss, ns_weighted, win, conv, ns_unweighted
    nb = 0
    for i in range(0, n, BATCH):
        idx = perm[i : i + BATCH]
        with torch.set_grad_enabled(train):
            out = model(
                data["trunk"][idx],
                data["wp"][idx],
                data["posteam"][idx],
                data["defteam"][idx],
            )

            # Next-score: weighted cross-entropy (nflfastR-style weights).
            ce = F.cross_entropy(
                out["next_score_logits"], data["next_score"][idx], reduction="none"
            )
            w = data["ep_weight"][idx]
            loss_ns = (ce * w).sum() / w.sum().clamp_min(1e-6)

            # Win: soft targets (ties are 0.5) handled natively by BCE.
            loss_win = F.binary_cross_entropy_with_logits(
                out["win_logit"], data["won"][idx]
            )

            # Convert: masked so only real 3rd/4th-down attempts contribute.
            m = data["convert_mask"][idx]
            conv_raw = F.binary_cross_entropy_with_logits(
                out["convert_logit"], data["converted"][idx], reduction="none"
            )
            loss_conv = (conv_raw * m).sum() / m.sum().clamp_min(1e-6)

            loss = W_NEXT_SCORE * loss_ns + W_WIN * loss_win + W_CONVERT * loss_conv

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        totals += np.array(
            [loss.item(), loss_ns.item(), loss_win.item(), loss_conv.item(), ce.mean().item()]
        )
        nb += 1

    return totals / nb


@torch.no_grad()
def predict(model, data, point_values, league_average: bool = False):
    model.eval()
    n = data["trunk"].shape[0]
    eps, wps, convs, ns_probs = [], [], [], []
    for i in range(0, n, 32768):
        sl = slice(i, i + 32768)
        out = model(
            data["trunk"][sl],
            data["wp"][sl],
            data["posteam"][sl],
            data["defteam"][sl],
            league_average=league_average,
        )
        p = torch.softmax(out["next_score_logits"], dim=1)
        ns_probs.append(p.cpu().numpy())
        eps.append((p @ point_values).cpu().numpy())
        wps.append(torch.sigmoid(out["win_logit"]).cpu().numpy())
        convs.append(torch.sigmoid(out["convert_logit"]).cpu().numpy())
    return (
        np.concatenate(eps),
        np.concatenate(wps),
        np.concatenate(convs),
        np.concatenate(ns_probs),
    )


def evaluate(name, df, data, ep_hat, wp_hat, conv_hat, ns_probs):
    print(f"\n--- {name} ({df.height:,} plays) ---")

    y_ns = data["next_score"].cpu().numpy()
    ll = log_loss(y_ns, ns_probs, labels=list(range(7)))
    acc = (ns_probs.argmax(1) == y_ns).mean()
    print(f"  next-score   log loss {ll:.4f}   top-1 acc {acc*100:5.2f}%")

    ep_ref = df["ep"].to_numpy()
    print(
        f"  EP vs nflfastR   r={np.corrcoef(ep_hat, ep_ref)[0,1]:.4f}   "
        f"MAE={np.abs(ep_hat-ep_ref).mean():.3f} pts   "
        f"bias={np.mean(ep_hat-ep_ref):+.3f}"
    )

    won = data["won"].cpu().numpy()
    decided = won != 0.5
    wp_ll = log_loss(won[decided], wp_hat[decided].clip(1e-6, 1 - 1e-6))
    wp_brier = brier_score_loss(won[decided], wp_hat[decided])
    wp_auc = roc_auc_score(won[decided], wp_hat[decided])
    print(f"  win prob     log loss {wp_ll:.4f}   Brier {wp_brier:.4f}   AUC {wp_auc:.4f}")

    # Two nflfastR references. `wp` ignores the betting market; `vegas_wp`
    # uses the pregame spread. Ours uses the spread too, so `vegas_wp` is the
    # like-for-like comparison and `wp` is the easier bar.
    for label, col in [("nflfastR wp ", "wp"), ("nflfastR vegas_wp", "vegas_wp")]:
        ref = df[col].to_numpy()
        ok = ~np.isnan(ref) & decided
        print(
            f"  {label} log loss {log_loss(won[ok], ref[ok].clip(1e-6,1-1e-6)):.4f}   "
            f"Brier {brier_score_loss(won[ok], ref[ok]):.4f}   "
            f"AUC {roc_auc_score(won[ok], ref[ok]):.4f}"
        )

    m = data["convert_mask"].cpu().numpy().astype(bool)
    yc = data["converted"].cpu().numpy()[m]
    pc = conv_hat[m]
    print(
        f"  conversion   log loss {log_loss(yc, pc.clip(1e-6,1-1e-6)):.4f}   "
        f"Brier {brier_score_loss(yc, pc):.4f}   AUC {roc_auc_score(yc, pc):.4f}   "
        f"(n={m.sum():,})"
    )


def reliability(name, y, p, bins=10):
    """Calibration table. For a probability model this matters more than AUC:
    a model can rank perfectly and still be systematically overconfident."""
    print(f"\n  {name} reliability")
    print(f"    {'bucket':<14}{'n':>8}{'predicted':>11}{'actual':>9}{'gap':>8}")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.sum() < 30:
            continue
        pred, act = p[sel].mean(), y[sel].mean()
        ece += sel.sum() / len(p) * abs(pred - act)
        print(
            f"    {lo:.1f}-{hi:.1f}{'':<6}{sel.sum():>8,}{pred:>11.3f}{act:>9.3f}{pred-act:>+8.3f}"
        )
    print(f"    expected calibration error = {ece:.4f}")


def main() -> None:
    ensure_dirs()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device()
    print(f"device: {dev}")

    df = pl.read_parquet(PROCESSED_DIR / "state_plays.parquet")
    train_df = df.filter(pl.col("season").is_in(TRAIN_SEASONS))
    val_df = df.filter(pl.col("season").is_in(VAL_SEASONS))
    test_df = df.filter(pl.col("season").is_in(TEST_SEASONS))
    print(
        f"train {train_df.height:,} | val {val_df.height:,} | test {test_df.height:,}"
    )

    pre = Preprocessor().fit(train_df)
    print(f"features: {pre.n_trunk} trunk + {pre.n_wp} wp | teams: {pre.n_teams}")
    if pre.dropped:
        print(
            f"  dropped (no variance in {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}): "
            f"{', '.join(pre.dropped)}"
        )
        print("  -> a rule change absent from training data cannot be learned.")

    # Class-prior baseline: any model that cannot beat this is not doing
    # anything. Printed every run so the headline loss is never read alone.
    prior = np.bincount(
        np.array([c for c in pre.transform(train_df)["next_score"]]), minlength=7
    ) / train_df.height
    val_y = pre.transform(val_df)["next_score"]
    base_ll = -np.mean(np.log(np.clip(prior[val_y], 1e-12, None)))
    print(f"  baseline next-score log loss (train class priors on val): {base_ll:.4f}")

    tr = to_tensors(pre.transform(train_df), dev)
    va = to_tensors(pre.transform(val_df), dev)
    te = to_tensors(pre.transform(test_df), dev)

    point_values = torch.tensor(NEXT_SCORE_VALUES, dtype=torch.float32, device=dev)

    model = MultiTaskStateNet(
        n_trunk_features=pre.n_trunk, n_wp_features=pre.n_wp, n_teams=pre.n_teams
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        trl = run_epoch(model, tr, opt, point_values, train=True)
        val = run_epoch(model, va, opt, point_values, train=False)
        sched.step()

        # Early stop on the two tasks we actually serve, using the UNWEIGHTED
        # next-score loss so the stopping criterion matches the metric we
        # report. Stopping on the weighted loss optimises a different
        # objective than the one being evaluated.
        score = val[4] + val[2]
        flag = ""
        if score < best - 1e-5:
            best, best_state, bad = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            flag = " *"
        else:
            bad += 1

        if ep % 5 == 0 or flag:
            print(
                f"  epoch {ep:>3}  train ns {trl[4]:.4f} win {trl[2]:.4f} conv {trl[3]:.4f}"
                f"  |  val ns {val[4]:.4f} win {val[2]:.4f} conv {val[3]:.4f}{flag}"
            )
        if bad >= PATIENCE:
            print(f"  early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    print(f"trained in {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    for name, d, data in [("VAL 2024", val_df, va), ("TEST 2025", test_df, te)]:
        ep_hat, wp_hat, conv_hat, ns_probs = predict(model, data, point_values)
        evaluate(name, d, data, ep_hat, wp_hat, conv_hat, ns_probs)

    # Calibration on the true holdout only.
    ep_hat, wp_hat, conv_hat, ns_probs = predict(model, te, point_values)
    won = te["won"].cpu().numpy()
    dec = won != 0.5
    reliability("TEST 2025 win probability", won[dec], wp_hat[dec])
    m = te["convert_mask"].cpu().numpy().astype(bool)
    reliability("TEST 2025 conversion", te["converted"].cpu().numpy()[m], conv_hat[m])

    torch.save(
        {"state_dict": model.state_dict(),
         "n_trunk": pre.n_trunk, "n_wp": pre.n_wp, "n_teams": pre.n_teams},
        MODEL_DIR / "state_net.pt",
    )
    pre.save(MODEL_DIR / "preprocessor")
    print(f"\nsaved model + preprocessor to {MODEL_DIR}")


if __name__ == "__main__":
    main()
