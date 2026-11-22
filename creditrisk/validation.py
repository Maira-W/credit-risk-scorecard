"""The measures a credit-risk team uses to judge a scoring model: how well it ranks, how honest its probabilities are, and
how steady it is. A "score" here is always a probability of default: higher means riskier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config

EPS = 1e-9


def auc(y, p) -> float:
    """Area under the ROC curve: the chance that a random bad account scores riskier than a random good one (ties count half)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    n_bad = int(y.sum())
    n_good = len(y) - n_bad
    if n_bad == 0 or n_good == 0:
        raise ValueError("both good and bad accounts are needed")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_good))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the share of bad and the share of good accounts at or below a score."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    _, inverse = np.unique(p, return_inverse=True)
    bad = np.bincount(inverse[y == 1], minlength=inverse.max() + 1)
    good = np.bincount(inverse[y == 0], minlength=inverse.max() + 1)
    return float(np.max(np.abs(np.cumsum(bad) / bad.sum() - np.cumsum(good) / good.sum())))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


METRICS = {"auc": auc, "gini": gini, "ks": ks, "brier": brier, "log_loss": log_loss}


def calibration_fit(y, p) -> tuple[float, float]:
    """Intercept and slope of a logistic regression of the outcome on the logit of the predicted probability. A model whose
    probabilities are right has intercept 0 and slope 1; a slope under 1 means its probabilities are too extreme."""
    import statsmodels.api as sm
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fit = sm.Logit(np.asarray(y), sm.add_constant(z)).fit(disp=0)
    return float(fit.params[0]), float(fit.params[1])


def calibration_table(y, p, bins: int = config.CALIBRATION_BINS) -> pd.DataFrame:
    """Accounts sorted by predicted probability and cut into equal groups: the average predicted and the observed default rate of each."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    frame["group"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    out = frame.groupby("group").agg(accounts=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "group"
    return out


def lift_table(y, p, bins: int = config.DECILES) -> pd.DataFrame:
    """Accounts sorted from the riskiest, cut into equal groups. Lift is a group's default rate over the portfolio's; capture is
    the share of all defaults found in the riskiest groups up to this one."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    frame["group"] = pd.qcut((-frame["p"]).rank(method="first"), bins, labels=False) + 1
    out = frame.groupby("group").agg(accounts=("y", "size"), defaults=("y", "sum"), mean_probability=("p", "mean"))
    out["default_rate"] = out["defaults"] / out["accounts"]
    out["lift"] = out["default_rate"] / frame["y"].mean()
    out["capture"] = out["defaults"].cumsum() / out["defaults"].sum()
    return out


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`: sum of (share now - share then) x ln(share now / share then) over
    bins cut at the deciles of `expected`. Under 0.10 is stable, 0.10 to 0.25 a shift worth a look, above 0.25 a large shift."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(cuts, expected, side="right"), minlength=len(cuts) + 1) / len(expected)
    a = np.bincount(np.searchsorted(cuts, actual, side="right"), minlength=len(cuts) + 1) / len(actual)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def roc_curve(y, p, max_points: int = 400) -> pd.DataFrame:
    """Points of the ROC curve (false-positive and true-positive rate), thinned to at most about `max_points`."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]                # the last row of each run of equal scores
    tpr = np.cumsum(ys)[last] / ys.sum()
    fpr = np.cumsum(1 - ys)[last] / (1 - ys).sum()
    curve = pd.DataFrame({"fpr": np.r_[0.0, fpr], "tpr": np.r_[0.0, tpr]})
    if len(curve) > max_points:
        curve = curve.iloc[np.unique(np.linspace(0, len(curve) - 1, max_points).astype(int))]
    return curve.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap(metric, y, scores: dict[str, np.ndarray], draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED) -> pd.DataFrame:
    """The metric of each score on `draws` resamples of the accounts (the same accounts for every score, so differences are paired).
    Returns one row per draw and one column per score."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {name: np.empty(draws) for name in scores}
    for d in range(draws):
        pick = rng.integers(0, n, n)
        ys = y[pick]
        if ys.min() == ys.max():                                           # cannot happen with real data; guards tiny tests
            for name in scores:
                out[name][d] = np.nan
            continue
        for name, p in scores.items():
            out[name][d] = metric(ys, np.asarray(p)[pick])
    return pd.DataFrame(out)


def interval(draws: pd.Series, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2
    return float(draws.quantile(tail)), float(draws.quantile(1 - tail))
