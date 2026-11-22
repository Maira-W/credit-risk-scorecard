"""A fairness audit of the decisions a model would make, by sex, age, education and marital status.

None of these is an input to the main models. They are used only here, to check what the decisions do to each group:
  * approval rate: the share of the group the cut-off would approve;
  * disparate-impact ratio: a group's approval rate over that of the best-treated group (the "four-fifths rule" flags under 0.80;
    it is a screening rule from US employment and lending practice, not a Pakistani legal test);
  * defaulters declined (true-positive rate) and good accounts declined (false-positive rate): who bears the model's mistakes;
  * calibration: the average predicted probability against the observed default rate, so that a model that is too harsh or too
    lenient on a group shows up. A model can be calibrated for every group and still approve groups at different rates when
    their default rates truly differ; the audit shows both so that the two are not confused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, validation


def group_table(frame: pd.DataFrame, attribute: str, cutoff: float, probability: str = "probability") -> pd.DataFrame:
    """One row per group of `attribute`. `frame` has the columns `default`, `probability` (of default) and the attribute;
    an account is approved when its probability is at or below `cutoff`."""
    approved_all = frame[probability] <= cutoff
    rows = []
    for level, part in frame.groupby(attribute, sort=True):
        approved = part[probability] <= cutoff
        bad, good = part["default"] == 1, part["default"] == 0
        rows.append({
            "attribute": attribute, "group": level, "accounts": len(part), "share": len(part) / len(frame),
            "default_rate": part["default"].mean(), "mean_probability": part[probability].mean(),
            "approval_rate": approved.mean(),
            "default_rate_approved": part.loc[approved, "default"].mean() if approved.any() else np.nan,
            "defaulters_declined": (~approved[bad]).mean() if bad.any() else np.nan,
            "good_declined": (~approved[good]).mean() if good.any() else np.nan,
            "auc": validation.auc(part["default"], part[probability]) if bad.any() and good.any() else np.nan,
        })
    table = pd.DataFrame(rows)
    table["calibration_ratio"] = table["mean_probability"] / table["default_rate"]
    judged = table["accounts"] >= config.MIN_GROUP_SIZE
    reference = table.loc[judged, "approval_rate"].max()
    table["disparate_impact"] = np.where(judged, table["approval_rate"] / reference if reference > 0 else np.nan, np.nan)
    table["flagged"] = judged & (table["disparate_impact"] < config.DISPARATE_IMPACT_MIN)
    table["approval_rate_overall"] = approved_all.mean()
    return table


def impact_interval(frame: pd.DataFrame, attribute: str, cutoff: float, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED,
                    probability: str = "probability") -> pd.DataFrame:
    """A 95% bootstrap range for each group's disparate-impact ratio: accounts are resampled, and in each resample every group's
    approval rate is divided by the highest among the groups that are big enough to be judged. A ratio whose whole range is under
    0.80 is flagged with more confidence than one whose range straddles it."""
    codes, levels = pd.factorize(frame[attribute], sort=True)
    approved = (frame[probability].to_numpy() <= cutoff).astype(float)
    k, n = len(levels), len(frame)
    judged = np.bincount(codes, minlength=k) >= config.MIN_GROUP_SIZE
    rng = np.random.default_rng(seed)
    ratios = np.full((draws, k), np.nan)
    for d in range(draws):
        pick = rng.integers(0, n, n)
        size = np.bincount(codes[pick], minlength=k)
        rate = np.bincount(codes[pick], weights=approved[pick], minlength=k) / np.maximum(size, 1)
        best = rate[judged].max() if judged.any() else 0.0
        if best > 0:
            ratios[d] = np.where(judged, rate / best, np.nan)
    low, high = np.full(k, np.nan), np.full(k, np.nan)
    if judged.any() and not np.isnan(ratios[:, judged]).all():
        low[judged] = np.nanpercentile(ratios[:, judged], 2.5, axis=0)
        high[judged] = np.nanpercentile(ratios[:, judged], 97.5, axis=0)
    return pd.DataFrame({"attribute": attribute, "group": levels, "di_low": low, "di_high": high})


def audit(frame: pd.DataFrame, attributes, cutoff: float, probability: str = "probability", draws: int = 0) -> pd.DataFrame:
    """The group tables of every attribute, stacked; with `draws` above 0, each judged group also gets a bootstrap range for its
    disparate-impact ratio (di_low, di_high)."""
    table = pd.concat([group_table(frame, a, cutoff, probability) for a in attributes], ignore_index=True)
    if draws:
        ranges = pd.concat([impact_interval(frame, a, cutoff, draws, probability=probability) for a in attributes], ignore_index=True)
        table = table.merge(ranges, on=["attribute", "group"], how="left")
    return table


def proxy_strength(X: pd.DataFrame, target, columns, folds: int = 3, seed: int = config.SEED, threads: int = config.CHALLENGER_THREADS) -> float:
    """How well the model's inputs can tell a protected attribute apart (cross-validated AUC of a gradient-boosting model):
    0.5 means the inputs carry no information about it; the nearer to 1, the more the inputs act as a stand-in for it."""
    import lightgbm as lgb

    from .models import stratified_folds
    target = np.asarray(target).astype(int)
    p = np.full(len(target), np.nan)
    for train, valid in stratified_folds(target, folds, seed):
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100, n_jobs=threads,
                                   random_state=seed, verbose=-1).fit(X.iloc[train][list(columns)], target[train])
        p[valid] = model.predict_proba(X.iloc[valid][list(columns)])[:, 1]
    return validation.auc(target, p)
