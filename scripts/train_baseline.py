"""XGBoost baseline for the three state-value tasks.

Run:  python scripts/train_baseline.py

Gradient boosting is what nflfastR itself uses, and on tabular data of this
size it is the model to beat -- not the model to feel clever for avoiding. The
point of this script is to find out honestly whether the multi-task network
earns its keep.

The structural difference: three separate boosters cannot share a
representation. Each learns its task from scratch. If the shared trunk is
doing real work, the network should show its biggest advantage on the
conversion task, which has by far the fewest labelled rows and therefore the
most to gain from borrowing.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fourthdown.config import (
    MODEL_DIR,
    PROCESSED_DIR,
    REPORT_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    ensure_dirs,
)
from fourthdown.preprocess import Preprocessor

SEED = 1738

COMMON = dict(
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=SEED,
    n_estimators=2000,
    early_stopping_rounds=50,
)


def matrices(pre: Preprocessor, df: pl.DataFrame) -> np.ndarray:
    """Same features the network sees, plus team indices.

    Trees do not need the standardization, but reusing the preprocessor
    guarantees both models get an identical feature set -- otherwise the
    comparison measures feature engineering rather than model class.
    """
    d = pre.transform(df)
    return np.column_stack(
        [d["trunk"], d["wp"], d["posteam"].astype(np.float32), d["defteam"].astype(np.float32)]
    )


def main() -> None:
    ensure_dirs()
    df = pl.read_parquet(PROCESSED_DIR / "state_plays.parquet")
    tr = df.filter(pl.col("season").is_in(TRAIN_SEASONS))
    va = df.filter(pl.col("season").is_in(VAL_SEASONS))
    te = df.filter(pl.col("season").is_in(TEST_SEASONS))

    pre = Preprocessor.load(MODEL_DIR / "preprocessor")
    Xtr, Xva, Xte = matrices(pre, tr), matrices(pre, va), matrices(pre, te)
    dtr, dva, dte = pre.transform(tr), pre.transform(va), pre.transform(te)
    print(f"features: {Xtr.shape[1]} | train {Xtr.shape[0]:,} val {Xva.shape[0]:,} test {Xte.shape[0]:,}")

    results: dict[str, dict] = {}

    # ---------------- next score (7-class) ----------------
    print("\n[1/3] next-score classifier ...")
    m_ns = xgb.XGBClassifier(objective="multi:softprob", num_class=7, **COMMON)
    m_ns.fit(
        Xtr, dtr["next_score"],
        sample_weight=dtr["ep_weight"],
        eval_set=[(Xva, dva["next_score"])],
        verbose=False,
    )
    print(f"   best iteration: {m_ns.best_iteration}")
    for name, X, d in [("VAL 2024", Xva, dva), ("TEST 2025", Xte, dte)]:
        p = m_ns.predict_proba(X)
        ll = log_loss(d["next_score"], p, labels=list(range(7)))
        acc = (p.argmax(1) == d["next_score"]).mean()
        ep = p @ np.array([7.0, 3.0, 2.0, 0.0, -2.0, -3.0, -7.0])
        ref = (va if name.startswith("VAL") else te)["ep"].to_numpy()
        print(
            f"   {name}: log loss {ll:.4f}  acc {acc*100:5.2f}%  "
            f"EP vs nflfastR r={np.corrcoef(ep, ref)[0,1]:.4f} MAE={np.abs(ep-ref).mean():.3f}"
        )
        results.setdefault(name, {})["next_score_logloss"] = float(ll)
        results[name]["next_score_acc"] = float(acc)
        results[name]["ep_mae"] = float(np.abs(ep - ref).mean())
        results[name]["ep_r"] = float(np.corrcoef(ep, ref)[0, 1])

    # ---------------- win probability ----------------
    print("\n[2/3] win-probability classifier ...")
    # Ties (label 0.5) are dropped here: XGBoost's logistic objective wants
    # 0/1 targets. They are 0.2% of plays.
    tr_dec = dtr["won"] != 0.5
    va_dec = dva["won"] != 0.5
    m_wp = xgb.XGBClassifier(objective="binary:logistic", **COMMON)
    m_wp.fit(
        Xtr[tr_dec], dtr["won"][tr_dec].astype(int),
        eval_set=[(Xva[va_dec], dva["won"][va_dec].astype(int))],
        verbose=False,
    )
    print(f"   best iteration: {m_wp.best_iteration}")
    for name, X, d in [("VAL 2024", Xva, dva), ("TEST 2025", Xte, dte)]:
        dec = d["won"] != 0.5
        p = m_wp.predict_proba(X[dec])[:, 1]
        y = d["won"][dec]
        print(
            f"   {name}: log loss {log_loss(y, p.clip(1e-6,1-1e-6)):.4f}  "
            f"Brier {brier_score_loss(y, p):.4f}  AUC {roc_auc_score(y, p):.4f}"
        )
        results[name]["wp_logloss"] = float(log_loss(y, p.clip(1e-6, 1 - 1e-6)))
        results[name]["wp_brier"] = float(brier_score_loss(y, p))
        results[name]["wp_auc"] = float(roc_auc_score(y, p))

    # ---------------- conversion ----------------
    print("\n[3/3] conversion classifier ...")
    mtr = dtr["convert_mask"].astype(bool)
    mva = dva["convert_mask"].astype(bool)
    m_cv = xgb.XGBClassifier(objective="binary:logistic", **COMMON)
    m_cv.fit(
        Xtr[mtr], dtr["converted"][mtr].astype(int),
        eval_set=[(Xva[mva], dva["converted"][mva].astype(int))],
        verbose=False,
    )
    print(f"   best iteration: {m_cv.best_iteration}   (train rows: {mtr.sum():,})")
    for name, X, d in [("VAL 2024", Xva, dva), ("TEST 2025", Xte, dte)]:
        m = d["convert_mask"].astype(bool)
        p = m_cv.predict_proba(X[m])[:, 1]
        y = d["converted"][m]
        print(
            f"   {name}: log loss {log_loss(y, p.clip(1e-6,1-1e-6)):.4f}  "
            f"Brier {brier_score_loss(y, p):.4f}  AUC {roc_auc_score(y, p):.4f}"
        )
        results[name]["conv_logloss"] = float(log_loss(y, p.clip(1e-6, 1 - 1e-6)))
        results[name]["conv_brier"] = float(brier_score_loss(y, p))
        results[name]["conv_auc"] = float(roc_auc_score(y, p))

    for m, n in [(m_ns, "xgb_next_score"), (m_wp, "xgb_win"), (m_cv, "xgb_convert")]:
        m.save_model(MODEL_DIR / f"{n}.json")
    (REPORT_DIR / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved boosters to {MODEL_DIR}, metrics to {REPORT_DIR}/baseline_metrics.json")


if __name__ == "__main__":
    main()
