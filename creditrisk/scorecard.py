"""The scorecard: weight-of-evidence features, a logistic regression, and the regression turned into points.

Fitting follows the usual bank workflow, using the development data only:
  1. bin every candidate feature and work out its information value (IV);
  2. drop features with an IV under `min_iv`;
  3. of two features whose weights of evidence correlate above `max_correlation`, keep the one with the higher IV;
  4. fit a logistic regression of default on the weights of evidence, then drop, one at a time, any feature whose coefficient
     is not positive (a bin that looks riskier must never make an account look safer) or whose p-value is above `max_p_value`;
  5. turn the regression into whole points: the score is SCORE_BASE at odds of SCORE_BASE_ODDS to 1 of good against bad, and each
     SCORE_PDO points double those odds. A higher score is a safer account.
The probability of default of an account is read from its whole-number score, so the card and the probability always agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import binning, config


@dataclass(frozen=True)
class Scaling:
    """The link between points and the probability of default: score = offset + factor x ln(odds of good against bad)."""

    base_score: float = config.SCORE_BASE
    base_odds: float = config.SCORE_BASE_ODDS
    pdo: float = config.SCORE_PDO

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score(self, probability) -> np.ndarray:
        p = np.asarray(probability, dtype=float)
        return self.offset + self.factor * np.log((1 - p) / p)

    def probability(self, score) -> np.ndarray:
        return 1.0 / (1.0 + np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor))


class Scorecard:
    def __init__(self, candidates=config.SCORECARD_CANDIDATES, *, min_iv: float = config.MIN_IV,
                 max_correlation: float = config.MAX_CORRELATION, max_p_value: float = config.MAX_P_VALUE,
                 min_share: float = config.MIN_BIN_SHARE, max_bins: int = config.MAX_BINS,
                 iv_tolerance: float = config.IV_TOLERANCE, scaling: Scaling | None = None, labels: dict | None = None):
        self.candidates = tuple(candidates)
        self.min_iv, self.max_correlation, self.max_p_value = min_iv, max_correlation, max_p_value
        self.min_share, self.max_bins, self.iv_tolerance = min_share, max_bins, iv_tolerance
        self.scaling = scaling or Scaling()
        self.labels = labels or {}

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y) -> Scorecard:
        y = np.asarray(y).astype(int)
        missing = [name for name, *_ in self.candidates if name not in X.columns]
        if missing:
            raise KeyError(f"the data have no column {missing[0]!r}")
        self.binnings_ = {name: binning.fit_binning(X[name], y, name, kind, monotone, min_share=self.min_share, max_bins=self.max_bins,
                                                    iv_tolerance=self.iv_tolerance, labels=self.labels.get(name))
                          for name, kind, monotone in self.candidates}
        notes = {name: "" for name in self.binnings_}

        ranked = sorted(self.binnings_, key=lambda n: -self.binnings_[n].iv)
        strong = [n for n in ranked if self.binnings_[n].iv >= self.min_iv and self.binnings_[n].n_bins >= 2]
        for n in ranked:
            if n not in strong:
                notes[n] = ("a single bin: no split of it separates good from bad accounts" if self.binnings_[n].n_bins < 2
                            else f"information value under {self.min_iv}")

        woe = pd.DataFrame({n: self.binnings_[n].woe(X[n]) for n in strong}, index=X.index)
        kept: list[str] = []
        for n in strong:
            clash = next((k for k in kept if abs(np.corrcoef(woe[n], woe[k])[0, 1]) > self.max_correlation), None)
            if clash is None:
                kept.append(n)
            else:
                notes[n] = f"correlated with {clash} (r = {np.corrcoef(woe[n], woe[clash])[0, 1]:.2f}), which has the higher information value"
        if not kept:
            raise ValueError("no feature has enough information value to build a scorecard")

        while True:
            result = sm.Logit(y, sm.add_constant(woe[kept], has_constant="add")).fit(disp=0, maxiter=200)
            coefficients, p_values = result.params.drop("const"), result.pvalues.drop("const")
            wrong_sign = coefficients[coefficients <= 0]
            if len(wrong_sign) and len(kept) > 1:
                worst = wrong_sign.idxmin()
                notes[worst] = f"coefficient {coefficients[worst]:.2f} is not positive once the others are in the model"
            elif p_values.max() > self.max_p_value and len(kept) > 1:
                worst = p_values.idxmax()
                notes[worst] = f"p-value {p_values[worst]:.3f} is above {self.max_p_value}"
            else:
                break
            kept.remove(worst)

        self.features_ = kept
        self.intercept_ = float(result.params["const"])
        self.coefficients_ = coefficients.astype(float)
        self.p_values_ = p_values.astype(float)
        self.screening_ = pd.DataFrame({
            "feature": ranked, "kind": [self.binnings_[n].kind for n in ranked], "bins": [self.binnings_[n].n_bins for n in ranked],
            "iv": [self.binnings_[n].iv for n in ranked],
            "decision": ["kept" if n in kept else "dropped" for n in ranked],
            "reason": [notes[n] for n in ranked],
            "coefficient": [self.coefficients_.get(n, np.nan) for n in ranked], "p_value": [self.p_values_.get(n, np.nan) for n in ranked],
        })

        scale, share = self.scaling, len(kept)
        self.points_, self.neutral_points_ = {}, {}
        for n in kept:
            w = self.binnings_[n].table["woe"].to_numpy()
            self.points_[n] = np.rint(-scale.factor * (self.coefficients_[n] * w + self.intercept_ / share) + scale.offset / share).astype(int)
            self.neutral_points_[n] = int(np.rint(-scale.factor * (self.intercept_ / share) + scale.offset / share))
        return self

    # ---------------------------------------------------------------- applying it
    def woe_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({n: self.binnings_[n].woe(X[n]) for n in self.features_}, index=X.index)

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """The points each feature gives each account."""
        out = {}
        for n in self.features_:
            table = np.append(self.points_[n], self.neutral_points_[n])        # index -1 (an unseen category) picks the neutral points
            out[n] = table[self.binnings_[n].bin_index(X[n])]
        return pd.DataFrame(out, index=X.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.points(X).sum(axis=1).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default read from the whole-number score."""
        return self.scaling.probability(self.score(X))

    def predict_proba_regression(self, X: pd.DataFrame) -> np.ndarray:
        """The regression's own probability, before the points were rounded."""
        z = self.intercept_ + self.woe_matrix(X).to_numpy() @ self.coefficients_[self.features_].to_numpy()
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------ reading it
    def points_table(self) -> pd.DataFrame:
        """The scorecard itself: every bin of every feature in the model, with its points."""
        frames = []
        for n in self.features_:
            t = self.binnings_[n].table
            frames.append(pd.DataFrame({"feature": n, "bin": t["label"], "accounts": t["n"], "default_rate": t["bad_rate"], "woe": t["woe"],
                                        "points": self.points_[n]}))
        return pd.concat(frames, ignore_index=True)

    def points_range(self) -> pd.Series:
        """How many points separate the best and the worst bin of each feature: its weight in the score."""
        return pd.Series({n: int(self.points_[n].max() - self.points_[n].min()) for n in self.features_}).sort_values(ascending=False)

    def reason_codes(self, X: pd.DataFrame, k: int = 3) -> list[list[tuple[str, str, int]]]:
        """For each account, the features that cost it most points against the best bin of that feature:
        (feature, the bin the account is in, points lost). Only features that cost points are listed."""
        points = self.points(X)
        lost = pd.DataFrame({n: self.points_[n].max() - points[n] for n in self.features_})
        labels = {n: self.binnings_[n].table["label"].to_numpy() for n in self.features_}
        index = {n: self.binnings_[n].bin_index(X[n]) for n in self.features_}
        out = []
        for row in range(len(X)):
            top = lost.iloc[row].sort_values(ascending=False).head(k)
            out.append([(n, labels[n][index[n][row]] if index[n][row] >= 0 else "unseen", int(v)) for n, v in top.items() if v > 0])
        return out
