"""The second layer: monotone checks, swap sets, stress tests, pricing, re-anchoring and fairness drivers, with answers worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import advanced, build, config, fairness, models, validation
from creditrisk.economics import Economics
from creditrisk.scorecard import Scaling

ECON = Economics(apr=0.44, revolve_share=0.9, funding=0.12, opex=0.05, loss=0.085)


class Wiggly:
    """A stand-in model: its probability follows the feature, going down and then up (so it breaks a rule that says "never down")."""

    columns = ["a", "b"]

    def predict_proba(self, frame):
        return 0.5 + 0.2 * np.sin(frame["a"].to_numpy() * 3)


class Steady:
    columns = ["a", "b"]

    def predict_proba(self, frame):
        return 1 / (1 + np.exp(-frame["a"].to_numpy()))


def test_the_monotone_check_counts_accounts_whose_probability_moves_the_wrong_way():
    frame = pd.DataFrame({"a": np.linspace(0, 3, 400), "b": np.arange(400)})
    wrong = advanced.monotone_violations(Wiggly(), frame, {"a": 1, "b": 1}, accounts=50).set_index("feature")
    assert wrong.loc["a", "violations"] == 50 and wrong.loc["b", "violations"] == 0
    assert 0.3 < wrong.loc["a", "largest_fall"] <= 0.4 and wrong.loc["a", "average_fall"] == pytest.approx(wrong.loc["a", "largest_fall"])     # the wave is 0.2 high: it can fall by at most 0.4
    assert wrong.loc["b", "largest_fall"] == 0 and wrong.loc["b", "average_fall"] == 0
    right = advanced.monotone_violations(Steady(), frame, {"a": 1}, accounts=50)
    assert right["violations"].sum() == 0 and right["largest_fall"].iloc[0] == 0
    falling = advanced.monotone_violations(Steady(), frame, {"a": -1}, accounts=50)
    assert falling["violations"].iloc[0] == 50
    assert falling["largest_fall"].iloc[0] == pytest.approx(1 / (1 + np.exp(-3)) - 0.5)                     # from a = 0 to a = 3 the probability rises by this much


def test_a_boosting_model_told_a_direction_obeys_it_and_does_no_worse_on_new_accounts():
    def draw(n, rng):
        x, z = rng.uniform(0, 1, n), rng.normal(size=n)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(3 * x - 1.5 + 0.3 * z)))).astype(int)        # the chance rises smoothly with x, plus noise
        return pd.DataFrame({"x": x, "z": z}), y

    rng = np.random.default_rng(0)
    train, y = draw(2000, rng)
    new, y_new = draw(8000, rng)
    params = {"n_estimators": 300, "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.05}
    free = models.Challenger(["x", "z"], params).fit(train, y)                                    # a small, noisy sample: the free model chases the noise
    told = models.Challenger(["x", "z"], params, monotone={"x": 1}).fit(train, y)
    assert advanced.monotone_violations(told, train, {"x": 1}, accounts=100)["violations"].iloc[0] == 0
    assert advanced.monotone_violations(free, train, {"x": 1}, accounts=100)["violations"].iloc[0] > 50
    assert validation.auc(y_new, told.predict_proba(new)) >= validation.auc(y_new, free.predict_proba(new))


def test_swap_sets_by_hand():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    balance = np.array([100.0, 100, 100, 100, 200, 200, 200, 200])
    a = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
    econ = Economics(apr=0.12 * 12, revolve_share=1.0, funding=0.0, opex=0.0, loss=0.5)           # margin 0.12 a month
    t = advanced.swap_sets(y, balance, a, b, econ, ("A", "B")).set_index("cell")
    assert list(t["accounts"]) == [2, 2, 2, 2] and t["share"].sum() == 1.0
    assert t.loc["Both approve", "default_rate"] == 0.5 and t.loc["Only A approves", "default_rate"] == 0.0
    assert t.loc["Only A approves", "profit_if_approved"] == pytest.approx((0.12 * 100 + 0.12 * 200) / 8)
    # the four cells add up to the profit of approving everyone
    everyone = (balance * np.where(y == 1, -0.5, 0.12)).sum() / 8
    assert t["profit_if_approved"].sum() == pytest.approx(everyone)


def test_shifting_the_odds():
    p = np.array([0.05, 0.2, 0.6])
    assert advanced.shift_odds(p, 1.0) == pytest.approx(p)
    shifted = advanced.shift_odds(p, 2.0)
    assert shifted / (1 - shifted) == pytest.approx(2 * p / (1 - p)) and ((shifted > p) & (shifted < 1)).all()
    assert advanced.shift_odds(p, 0.5).max() < p.max()


def test_expected_profit_matches_the_formula():
    p, e = np.array([0.1, 0.3]), np.array([1000.0, 2000.0])
    both = advanced.expected_profit(p, e, ECON, [True, True])
    by_hand = (1000 * ((0.9) * ECON.margin - 0.1 * ECON.loss) + 2000 * ((0.7) * ECON.margin - 0.3 * ECON.loss)) / 2
    assert both == pytest.approx(by_hand) and advanced.expected_profit(p, e, ECON, [False, False]) == 0


def test_stress_test_base_case_and_direction():
    rng = np.random.default_rng(1)
    p = np.clip(rng.beta(2, 6, 3000), 0.01, 0.9)
    e = rng.uniform(1000, 5000, 3000)
    policy = pd.DataFrame({"effective_date": pd.to_datetime(["2021-10-01", "2023-06-27"]), "policy_rate_pct": [7.25, 22.0]})
    scenarios = (("Base", 1.0, None), ("Defaults +50%", 1.5, None), ("Defaults x2", 2.0, None), ("Peak rate", 1.0, "2023-06-27"), ("Peak and +50%", 1.5, "2023-06-27"))
    t = advanced.stress_table(p, e, ECON, scenarios, policy).set_index("scenario")
    assert t.loc["Base", "profit_kept"] == pytest.approx(t.loc["Base", "profit_reset"]) and t.loc["Base", "approved_kept"] == pytest.approx(t.loc["Base", "approved_reset"])
    assert t.loc["Base", "profit_kept"] > t.loc["Defaults +50%", "profit_kept"] > t.loc["Defaults x2", "profit_kept"]
    assert (t["profit_reset"] >= t["profit_kept"] - 1e-9).all()                         # re-setting never does worse in expectation
    assert t.loc["Defaults x2", "approved_reset"] < t.loc["Base", "approved_reset"] and t.loc["Peak rate", "break_even_pd"] < t.loc["Base", "break_even_pd"]
    assert t.loc["Peak rate", "policy_rate"] == pytest.approx(0.22) and t.loc["Base", "policy_rate"] == pytest.approx(ECON.funding)


def test_the_pricing_table_prices_each_group_to_break_even():
    rng = np.random.default_rng(2)
    p = np.clip(rng.beta(2, 6, 5000), 0.01, 0.95)
    e = rng.uniform(1000, 5000, 5000)
    t = advanced.pricing_table(p, e, ECON, groups=10)
    assert len(t) == 10 and t["accounts"].sum() == 5000 and t["mean_probability"].is_monotonic_increasing and t["required_apr"].is_monotonic_increasing
    for row in t.itertuples():
        priced = ECON.with_(apr=row.required_apr)
        assert (1 - row.mean_probability) * priced.margin - row.mean_probability * ECON.loss == pytest.approx(0, abs=1e-12)       # worth exactly nothing at that APR
        assert row.pays_at_current_apr == (row.required_apr <= ECON.apr)


def test_logit_and_expit_are_inverses():
    p = np.array([0.001, 0.1, 0.5, 0.9])
    assert advanced.expit(advanced.logit(p)) == pytest.approx(p)


def test_re_anchoring_to_the_same_rate_changes_nothing_and_a_lower_rate_approves_more():
    rng = np.random.default_rng(3)
    p = np.clip(rng.beta(2, 6, 4000), 0.01, 0.95)
    e = rng.uniform(1000, 5000, 4000)
    t = advanced.local_rate_table(p, e, ECON, Scaling(), [0.05, 0.10], from_rate=0.22).set_index("local_default_rate")
    same = t.loc[0.22]
    assert same["log_odds_shift"] == pytest.approx(0) and same["approved_share"] == pytest.approx((p <= ECON.break_even).mean())
    assert same["cutoff_probability_in_original_terms"] == pytest.approx(ECON.break_even)
    assert t.loc[0.05, "approved_share"] > t.loc[0.10, "approved_share"] > same["approved_share"]
    assert t.loc[0.05, "cutoff_score"] < t.loc[0.10, "cutoff_score"] < same["cutoff_score"]           # a safer book can approve at fewer points
    assert t.loc[0.05, "expected_profit_per_account"] > same["expected_profit_per_account"]


def test_point_gaps_add_up_to_the_gap_in_the_average_score():
    rng = np.random.default_rng(5)
    points = pd.DataFrame({"f1": rng.integers(30, 70, 600), "f2": rng.integers(20, 60, 600), "f3": rng.integers(40, 50, 600)})
    groups = pd.Series(np.where(np.arange(600) % 3 == 0, "A", np.where(np.arange(600) % 3 == 1, "B", "C")))
    gaps = fairness.point_gaps(points, groups, "A", ["B", "C"])
    total = points.sum(axis=1)
    for group in ("B", "C"):
        part = gaps[gaps["group"] == group]
        assert part["points_gap"].sum() == pytest.approx(total[groups == group].mean() - total[groups == "A"].mean())
        assert part["total_points_gap"].iloc[0] == pytest.approx(part["points_gap"].sum()) and set(part["feature"]) == {"f1", "f2", "f3"}


def test_the_monotone_settings_name_real_features_and_never_a_repayment_code():
    assert set(config.MONOTONE) <= set(build.BEHAVIOUR) and set(config.MONOTONE.values()) == {-1, 1}
    assert not {c for c in config.MONOTONE if c.startswith("status_")}                    # a repayment status is a code, not a quantity
