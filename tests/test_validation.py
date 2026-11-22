"""The validation measures, against a library, hand calculations and made-up data with known properties."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics import log_loss as sk_log_loss

from creditrisk import validation as v


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(2, 6, 4000), 0.001, 0.999)
    y = (rng.uniform(size=4000) < p).astype(int)
    return y, p


def test_auc_matches_scikit_learn_including_ties(sample):
    y, p = sample
    assert v.auc(y, p) == pytest.approx(roc_auc_score(y, p))
    tied = np.round(p, 1)
    assert v.auc(y, tied) == pytest.approx(roc_auc_score(y, tied))


def test_auc_of_perfect_reversed_and_useless_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert v.auc(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == 1.0
    assert v.auc(y, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == 0.0
    assert v.auc(y, np.full(6, 0.5)) == 0.5
    assert v.gini(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == 1.0


def test_ks_by_hand_and_against_scipy(sample):
    assert v.ks([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5)
    y, p = sample
    assert v.ks(y, p) == pytest.approx(ks_2samp(p[y == 1], p[y == 0]).statistic)
    assert v.ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_brier_and_log_loss_match_scikit_learn(sample):
    y, p = sample
    assert v.brier(y, p) == pytest.approx(brier_score_loss(y, p))
    assert v.log_loss(y, p) == pytest.approx(sk_log_loss(y, p))


def test_a_score_of_exactly_zero_or_one_does_not_break_log_loss():
    assert np.isfinite(v.log_loss([0, 1], [1.0, 0.0]))


def test_one_class_only_is_an_error():
    with pytest.raises(ValueError):
        v.auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_calibrated_probabilities_have_intercept_zero_and_slope_one(sample):
    y, p = sample
    intercept, slope = v.calibration_fit(y, p)
    assert intercept == pytest.approx(0, abs=0.15)
    assert slope == pytest.approx(1, abs=0.12)


def test_overconfident_probabilities_have_a_slope_under_one(sample):
    y, p = sample
    z = np.log(p / (1 - p))
    too_extreme = 1 / (1 + np.exp(-2 * z))
    assert v.calibration_fit(y, too_extreme)[1] == pytest.approx(0.5, abs=0.1)


def test_calibration_table(sample):
    y, p = sample
    table = v.calibration_table(y, p, bins=10)
    assert len(table) == 10 and table["accounts"].sum() == len(y)
    assert (table["observed"] * table["accounts"]).sum() == pytest.approx(y.sum())
    assert (np.diff(table["predicted"]) > 0).all()


def test_lift_table(sample):
    y, p = sample
    table = v.lift_table(y, p, bins=10)
    assert table["defaults"].sum() == y.sum()
    assert table["capture"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > 1 > table["lift"].iloc[-1]
    assert (np.diff(table["default_rate"]) <= 1e-9).sum() >= 7        # riskier groups first, with a little noise allowed


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20_000)
    assert v.psi(a, rng.normal(0, 1, 20_000)) < 0.01
    assert v.psi(a, rng.normal(1.5, 1, 20_000)) > 0.25
    assert v.psi(a, a) == pytest.approx(0, abs=1e-9)


def test_roc_curve_runs_corner_to_corner_and_its_area_is_the_auc(sample):
    y, p = sample
    curve = v.roc_curve(y, p, max_points=100_000)
    assert (curve.iloc[0] == 0).all() and (curve.iloc[-1] == 1).all()
    assert (np.diff(curve["fpr"]) >= 0).all() and (np.diff(curve["tpr"]) >= 0).all()
    assert np.trapezoid(curve["tpr"], curve["fpr"]) == pytest.approx(v.auc(y, p), abs=1e-6)
    assert len(v.roc_curve(y, p, max_points=50)) <= 51


def test_bootstrap_intervals_cover_the_estimate_and_differences_are_paired(sample):
    y, p = sample
    draws = v.bootstrap(v.auc, y, {"a": p, "same": p, "worse": 1 - p}, draws=200, seed=3)
    low, high = v.interval(draws["a"])
    assert low < v.auc(y, p) < high
    assert ((draws["a"] - draws["same"]) == 0).all()                  # the same accounts are drawn for every score
    assert (draws["worse"] < draws["a"]).all()
    assert isinstance(draws, pd.DataFrame) and len(draws) == 200
