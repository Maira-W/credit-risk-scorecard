"""The data split, the naive baseline, the gradient-boosting challenger, and cross-validation.

The 30,000 accounts are split once into a development set (80%) and a hold-out (20%). Everything is chosen on the development set:
the bins, the features, the settings of the challenger, the cut-off. The hold-out is used once, at the end, to report results.
Accounts with identical features always stay on the same side of a split, so no account is tested on its own twin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config, validation


def holdout_split(accounts: pd.DataFrame, share: float = config.HOLDOUT_SHARE, seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the development set and of the hold-out. Stratified on the default flag and grouped on identical accounts."""
    folds = round(1 / share)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    development, holdout = next(splitter.split(accounts, accounts["default"], accounts["group"]))
    return development, holdout


def cv_splits(y, groups, folds: int, repeats: int, seed: int = config.SEED):
    """(repeat, fold, train positions, validation positions) for repeated stratified, grouped k-fold cross-validation."""
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, valid) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            yield repeat, fold, train, valid


# ---------------------------------------------------------------------------
# Models. Each has fit(X, y) and predict_proba(X) on a table of accounts, and the probability it returns is of default.
# ---------------------------------------------------------------------------
class RateByStatus:
    """The naive baseline: an account's chance of default is the default rate of the accounts with the same repayment status last month."""

    def __init__(self, column: str = "status_sep"):
        self.column = column

    def fit(self, X: pd.DataFrame, y) -> RateByStatus:
        y = np.asarray(y)
        self.rates_ = pd.Series(y).groupby(X[self.column].to_numpy()).mean()
        self.overall_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].map(self.rates_).fillna(self.overall_).to_numpy(dtype=float)


class Platt:
    """Logistic recalibration of probabilities: p becomes sigmoid(a + b x logit(p)), with a and b fitted on out-of-fold predictions.
    A model whose probabilities are a little too extreme has b under 1; this pulls them back without changing the ranking."""

    def fit(self, p, y) -> Platt:
        self.intercept_, self.slope_ = validation.calibration_fit(y, p)
        return self

    def predict(self, p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-(self.intercept_ + self.slope_ * np.log(p / (1 - p)))))


class Challenger:
    """Gradient-boosted trees (LightGBM) on the columns it is given."""

    def __init__(self, columns, params: dict | None = None, seed: int = config.SEED, threads: int = config.CHALLENGER_THREADS):
        self.columns = list(columns)
        self.params = {**config.CHALLENGER_PARAMS, **(params or {})}
        self.seed, self.threads = seed, threads

    def fit(self, X: pd.DataFrame, y) -> Challenger:
        import lightgbm as lgb
        self.model_ = lgb.LGBMClassifier(**self.params, n_jobs=self.threads, random_state=self.seed, verbose=-1,
                                         deterministic=True, force_row_wise=True)
        self.model_.fit(X[self.columns], np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns])[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values: how much each column pushed each account's log-odds of default up or down, against the average account."""
        values = self.model_.predict(X[self.columns], pred_contrib=True)
        return pd.DataFrame(values[:, :-1], columns=self.columns, index=X.index)


# ---------------------------------------------------------------------------
# Cross-validation and tuning
# ---------------------------------------------------------------------------
def cross_validate(factories: dict, X: pd.DataFrame, y, groups, folds: int = config.CV_FOLDS, repeats: int = config.CV_REPEATS,
                   seed: int = config.SEED) -> pd.DataFrame:
    """Every model fitted from scratch on each training part (binning and feature choice included) and scored on the validation part."""
    y = np.asarray(y)
    rows = []
    for repeat, fold, train, valid in cv_splits(y, groups, folds, repeats, seed):
        for name, make in factories.items():
            p = make().fit(X.iloc[train], y[train]).predict_proba(X.iloc[valid])
            rows.append({"model": name, "repeat": repeat, "fold": fold, **{m: f(y[valid], p) for m, f in validation.METRICS.items()}})
    return pd.DataFrame(rows)


def out_of_fold(make, X: pd.DataFrame, y, groups, folds: int = config.CV_FOLDS, seed: int = config.SEED) -> np.ndarray:
    """A prediction for every account from a model that did not see it: one round of k-fold cross-validation."""
    y = np.asarray(y)
    predictions = np.full(len(y), np.nan)
    for _, _, train, valid in cv_splits(y, groups, folds, 1, seed):
        predictions[valid] = make().fit(X.iloc[train], y[train]).predict_proba(X.iloc[valid])
    return predictions


SEARCH_SPACE = {
    "num_leaves": [7, 15, 31], "min_child_samples": [50, 100, 200], "learning_rate": [0.02, 0.03, 0.05],
    "n_estimators": [300, 500, 800], "colsample_bytree": [0.5, 0.7, 0.9], "reg_lambda": [1.0, 5.0, 20.0], "subsample": [0.7, 0.8, 0.9],
}


def tune_challenger(X: pd.DataFrame, y, groups, columns, trials: int = config.TUNING_TRIALS, folds: int = config.CV_FOLDS,
                    seed: int = config.SEED) -> tuple[dict, pd.DataFrame]:
    """Random search: `trials` settings drawn from SEARCH_SPACE, each judged by cross-validated log loss on the development set.
    The settings in config.CHALLENGER_PARAMS are always one of the candidates. Returns the best settings and every trial."""
    rng = np.random.default_rng(seed)
    candidates = [dict(config.CHALLENGER_PARAMS)]
    for _ in range(max(trials - 1, 0)):
        candidates.append({**config.CHALLENGER_PARAMS, **{k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}})
    y = np.asarray(y)
    rows = []
    for number, params in enumerate(candidates):
        losses, aucs = [], []
        for _, _, train, valid in cv_splits(y, groups, folds, 1, seed):
            p = Challenger(columns, params, seed=seed).fit(X.iloc[train], y[train]).predict_proba(X.iloc[valid])
            losses.append(validation.log_loss(y[valid], p))
            aucs.append(validation.auc(y[valid], p))
        rows.append({"trial": number, **params, "cv_log_loss": float(np.mean(losses)), "cv_auc": float(np.mean(aucs))})
    table = pd.DataFrame(rows)
    best = table.sort_values(["cv_log_loss", "trial"]).iloc[0]
    return {k: (int(best[k]) if k in ("num_leaves", "min_child_samples", "n_estimators") else float(best[k])) for k in SEARCH_SPACE}, table


def stratified_folds(y, folds: int, seed: int = config.SEED):
    """Plain stratified k-fold, for the small side checks that have no duplicate groups to keep together."""
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
