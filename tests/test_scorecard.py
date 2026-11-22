"""The scorecard, on accounts whose default follows a known logistic model."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import scorecard, validation
from creditrisk.scorecard import Scaling, Scorecard

CANDIDATES = (("x1", "numeric", None), ("x2", "numeric", None), ("x3", "numeric", None), ("x4", "numeric", None), ("x5", "numeric", None))


def test_the_base_score_is_at_the_base_odds():
    s = Scaling(base_score=600, base_odds=20, pdo=40)
    assert s.probability(600) == pytest.approx(1 / 21)
    assert s.score(1 / 21) == pytest.approx(600)


def test_every_pdo_points_double_the_odds_of_a_good_account():
    s = Scaling(base_score=600, base_odds=20, pdo=40)
    odds = lambda score: (1 - s.probability(score)) / s.probability(score)  # noqa: E731
    assert odds(640) == pytest.approx(2 * odds(600))
    assert odds(560) == pytest.approx(odds(600) / 2)


def test_score_and_probability_are_inverses_and_riskier_means_fewer_points():
    s = Scaling()
    p = np.array([0.02, 0.1, 0.3, 0.8])
    assert s.probability(s.score(p)) == pytest.approx(p)
    assert (np.diff(s.score(p)) < 0).all()


@pytest.fixture(scope="module")
def fitted(logistic_world):
    X, y, p = logistic_world
    train = np.arange(len(X)) < 14_000
    card = Scorecard(CANDIDATES).fit(X[train], y[train])
    return card, X[~train].reset_index(drop=True), y[~train], p[~train]


def test_it_keeps_the_features_that_matter_and_drops_the_rest(fitted):
    card, *_ = fitted
    assert set(card.features_) == {"x1", "x2"}
    reasons = card.screening_.set_index("feature")["reason"]
    assert "information value" in reasons["x3"]
    assert "correlated with x1" in reasons["x4"]


def test_every_coefficient_is_positive_and_significant(fitted):
    card, *_ = fitted
    assert (card.coefficients_ > 0).all()
    assert (card.p_values_ <= card.max_p_value).all()


def test_it_ranks_nearly_as_well_as_the_true_model(fitted):
    card, X, y, p_true = fitted
    assert validation.auc(y, card.predict_proba(X)) > validation.auc(y, p_true) - 0.02


def test_probabilities_are_calibrated_on_new_accounts(fitted):
    card, X, y, _ = fitted
    p = card.predict_proba(X)
    assert p.mean() == pytest.approx(y.mean(), abs=0.015)
    intercept, slope = validation.calibration_fit(y, p)
    assert slope == pytest.approx(1.0, abs=0.12)


def test_the_score_is_the_sum_of_whole_number_points(fitted):
    card, X, *_ = fitted
    points = card.points(X)
    assert all(np.issubdtype(points[c].dtype, np.integer) for c in points)
    assert (card.score(X) == points.sum(axis=1)).all()
    assert card.predict_proba(X) == pytest.approx(card.scaling.probability(card.score(X)))


def test_a_higher_score_always_means_a_lower_probability(fitted):
    card, X, *_ = fitted
    frame = pd.DataFrame({"score": card.score(X), "p": card.predict_proba(X)}).drop_duplicates().sort_values("score")
    assert (np.diff(frame["p"]) < 0).all()


def test_rounding_the_points_costs_almost_nothing(fitted):
    card, X, y, _ = fitted
    assert abs(validation.auc(y, card.predict_proba(X)) - validation.auc(y, card.predict_proba_regression(X))) < 0.003


def test_the_points_table_lists_every_bin_of_every_kept_feature(fitted):
    card, *_ = fitted
    table = card.points_table()
    assert len(table) == sum(card.binnings_[f].n_bins for f in card.features_)
    assert set(table["feature"]) == set(card.features_)
    # riskier bins (higher default rate) never get more points within a feature
    for _, part in table.groupby("feature"):
        assert (np.sign(np.diff(part.sort_values("woe")["points"])) <= 0).all()


def test_points_range_ranks_the_stronger_feature_first(fitted):
    card, *_ = fitted
    assert card.points_range().index[0] == "x1"


def test_reason_codes_name_the_worst_bins(fitted):
    card, X, *_ = fitted
    worst = X.loc[[int(np.argmin(card.score(X)))]]
    best = X.loc[[int(np.argmax(card.score(X)))]]
    reasons = card.reason_codes(worst, k=2)[0]
    assert 1 <= len(reasons) <= 2
    assert all(points > 0 for _, _, points in reasons)
    assert reasons == sorted(reasons, key=lambda r: -r[2])
    assert card.reason_codes(best, k=2)[0] == [] or all(p <= 5 for _, _, p in card.reason_codes(best, k=2)[0])


def test_features_with_no_information_stay_out_even_when_the_information_filter_is_switched_off(logistic_world):
    X, y, _ = logistic_world
    card = Scorecard(CANDIDATES, min_iv=0.0, max_correlation=0.99).fit(X, y)      # no IV floor: x3 and x5 are noise
    reasons = card.screening_.set_index("feature")["reason"]
    for noise in ("x3", "x5"):
        assert noise not in card.features_
        assert reasons[noise] != ""


def test_fitting_twice_gives_the_same_card(logistic_world):
    X, y, _ = logistic_world
    a, b = Scorecard(CANDIDATES).fit(X, y), Scorecard(CANDIDATES).fit(X, y)
    pd.testing.assert_frame_equal(a.points_table(), b.points_table())


def test_a_missing_column_is_reported_by_name(logistic_world):
    X, y, _ = logistic_world
    with pytest.raises(KeyError, match="x1"):
        Scorecard(CANDIDATES).fit(X.drop(columns="x1"), y)


def test_no_feature_with_information_is_an_error():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=3000)})
    with pytest.raises(ValueError, match="information value"):
        Scorecard((("x1", "numeric", None),)).fit(X, rng.integers(0, 2, 3000))


def test_module_exposes_the_scaling_used_by_default():
    assert scorecard.Scaling().base_score == 600
