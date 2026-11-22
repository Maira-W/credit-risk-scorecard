"""The profit of a cut-off, checked by hand and against the theory that the best cut-off is the break-even probability."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import config, economics
from creditrisk.economics import Economics

Y = np.array([0, 1, 0, 0])
P = np.array([0.1, 0.5, 0.2, 0.1])
BALANCE = np.array([100.0, 200.0, 300.0, 400.0])
ECON = Economics(apr=0.0, funding=0.0, opex=0.0, loss=0.5)           # zero margin placeholder, replaced below


def econ(margin_month: float, loss: float) -> Economics:
    """An Economics whose monthly margin is exactly `margin_month` (all of it from the APR)."""
    return Economics(apr=margin_month * 12, revolve_share=1.0, funding=0.0, opex=0.0, loss=loss)


def test_margin_and_break_even():
    e = Economics(apr=0.44, revolve_share=0.9, funding=0.12, opex=0.05, loss=0.15)
    assert e.yield_ == pytest.approx(0.396)
    assert e.margin == pytest.approx((0.396 - 0.12 - 0.05) / 12)
    assert e.break_even == pytest.approx(e.margin / (e.margin + 0.15))


def test_no_margin_means_no_approvals_at_all():
    e = Economics(apr=0.10, funding=0.12, opex=0.05)
    assert e.margin < 0 and e.break_even == 0.0


def test_changing_one_assumption_leaves_the_others():
    e = Economics().with_(loss=0.3)
    assert e.loss == 0.3 and e.apr == Economics().apr


def test_higher_funding_cost_lowers_the_margin_and_the_cutoff():
    low, high = Economics(funding=0.0725), Economics(funding=0.22)
    assert high.margin < low.margin and high.break_even < low.break_even


def test_threshold_table_sums():
    t = economics.threshold_table(Y, P, BALANCE)
    assert list(t["probability"]) == [0.1, 0.2, 0.5]
    assert list(t["accounts"]) == [2, 1, 1]
    assert list(t["balance_paid"]) == [500.0, 300.0, 0.0]
    assert list(t["balance_defaulted"]) == [0.0, 0.0, 200.0]
    assert t["balance_x_probability"].sum() == pytest.approx((P * BALANCE).sum())


def test_profit_curve_by_hand():
    e = econ(0.02, 0.5)
    curve = economics.profit_curve(economics.threshold_table(Y, P, BALANCE), e)
    # approve up to 0.1: balances 500 of good accounts earn 0.02 x 500; up to 0.2 add 300; up to 0.5 add the 200 that defaults
    assert list(curve["realised_profit"]) == pytest.approx([10.0, 16.0, 16.0 - 100.0])
    assert list(curve["approved_share"]) == [0.5, 0.75, 1.0]
    assert curve["expected_profit"].iloc[-1] == pytest.approx(0.02 * (1000 - (P * BALANCE).sum()) - 0.5 * (P * BALANCE).sum())
    best = economics.best_cutoff(curve)
    assert best["cutoff"] == 0.2 and best["realised_profit"] == pytest.approx(16.0)


def test_profit_at_a_cutoff_matches_the_curve():
    e = econ(0.02, 0.5)
    t = economics.threshold_table(Y, P, BALANCE)
    result = economics.profit_at(t, e, 0.2)
    assert result["realised_profit"] == pytest.approx(16.0) and result["approved_share"] == 0.75
    assert result["default_rate_approved"] == 0.0
    nobody = economics.profit_at(t, e, 0.05)
    assert nobody["approved_share"] == 0 and nobody["realised_profit"] == 0 and np.isnan(nobody["default_rate_approved"])


def test_when_every_cutoff_loses_money_nobody_is_approved():
    e = econ(0.0001, 0.9)
    curve = economics.profit_curve(economics.threshold_table(np.array([1, 1, 0]), np.array([0.5, 0.6, 0.7]), np.array([10.0, 10, 10])), e)
    best = economics.best_cutoff(curve)
    assert best["realised_profit"] == 0 and best["approved_share"] == 0 and best["cutoff"] == -np.inf


def test_the_best_cutoff_is_the_break_even_probability_when_probabilities_are_right():
    rng = np.random.default_rng(11)
    n = 400_000
    p = rng.beta(2, 6, n)
    y = (rng.uniform(size=n) < p).astype(int)
    e = econ(0.0183, 0.15)
    table = economics.threshold_table(y, np.round(p, 3), np.ones(n))
    best = economics.best_cutoff(economics.profit_curve(table, e))
    assert e.break_even == pytest.approx(0.1086, abs=1e-3)
    assert best["cutoff"] == pytest.approx(e.break_even, abs=0.02)


def test_coarsen_keeps_totals_and_makes_rows_of_at_least_the_minimum_size():
    rng = np.random.default_rng(2)
    n = 5000
    p = np.round(rng.beta(2, 6, n), 3)
    y = (rng.uniform(size=n) < p).astype(int)
    t = economics.threshold_table(y, p, rng.uniform(0, 100, n))
    c = economics.coarsen(t, 0.05)
    for column in ("accounts", "defaults", "balance_paid", "balance_defaulted", "balance"):
        assert c[column].sum() == pytest.approx(t[column].sum())
    assert (c["accounts"].iloc[:-1] >= 0.05 * n).all()
    assert c["probability"].is_monotonic_increasing and list(c.columns) == list(t.columns)
    full, coarse = economics.profit_curve(t, econ(0.02, 0.2)), economics.profit_curve(c, econ(0.02, 0.2))
    assert coarse["realised_profit"].iloc[-1] == pytest.approx(full["realised_profit"].iloc[-1])


def test_revolve_share_by_hand():
    months = config.MONTHS
    df = pd.DataFrame({f"bill_{m}": [1000, 2000] for m in months} | {f"paid_{m}": [0, 0] for m in months})
    assert economics.revolve_share(df) == 1.0                              # nothing was paid
    for m in months[:-1]:
        df[f"paid_{m}"] = [1000, 2000]
    assert economics.revolve_share(df) == 0.0                              # every bill was paid in full
    df["paid_sep"] = [500, 0]                                              # one of five months: 500 of 3000 paid
    other = [f"paid_{m}" for m in months[1:-1]]
    df[other] = 0
    assert economics.revolve_share(df) == pytest.approx(1 - 500 / (5 * 3000))


def test_exposure_is_the_september_balance_never_below_zero():
    df = pd.DataFrame({"bill_sep": [-50, 0, 700]})
    assert list(economics.exposure(df)) == [0.0, 0.0, 700.0]


def test_policy_rate_on_a_date():
    table = pd.DataFrame({"effective_date": pd.to_datetime(["2021-10-01", "2023-06-27"]), "policy_rate_pct": [7.25, 22.0]})
    assert economics.policy_rate_on("2021-10-01", table) == pytest.approx(0.0725)
    assert economics.policy_rate_on("2022-12-31", table) == pytest.approx(0.0725)
    assert economics.policy_rate_on("2023-06-27", table) == pytest.approx(0.22)
    real = economics.policy_rate_on("2026-11-22")
    assert 0.05 < real < 0.25


def test_scenarios_follow_the_policy_rate():
    rng = np.random.default_rng(5)
    n = 20_000
    p = np.round(rng.beta(2, 6, n), 3)
    y = (rng.uniform(size=n) < p).astype(int)
    table = economics.threshold_table(y, p, rng.uniform(1, 100, n))
    policy = pd.DataFrame({"effective_date": pd.to_datetime(["2021-10-01", "2023-06-27", "2025-01-01"]), "policy_rate_pct": [7.25, 22.0, 11.0]})
    s = economics.scenarios(table, Economics(), ["2021-10-01", "2023-06-27", "2025-02-01"], policy)
    assert list(s["policy_rate"]) == pytest.approx([0.0725, 0.22, 0.11])
    assert s["break_even_pd"].iloc[1] < s["break_even_pd"].iloc[2] < s["break_even_pd"].iloc[0]
    assert s["approved_share"].iloc[1] < s["approved_share"].iloc[2] < s["approved_share"].iloc[0]
    assert s["profit_per_account"].iloc[0] == pytest.approx(s["profit_per_account_first_cutoff_kept"].iloc[0])
    assert s["profit_per_account_first_cutoff_kept"].iloc[1] < s["profit_per_account"].iloc[1]      # keeping the 2021 cut-off in 2023 costs money


def test_sensitivity_covers_every_combination():
    rng = np.random.default_rng(6)
    p = np.round(rng.beta(2, 6, 5000), 3)
    table = economics.threshold_table((rng.uniform(size=5000) < p).astype(int), p, np.ones(5000))
    s = economics.sensitivity(table, Economics(), [0.1, 0.2], [0.3, 0.44])
    assert len(s) == 4 and set(s["apr"]) == {0.3, 0.44}
    assert (s.groupby("apr")["break_even_pd"].apply(lambda v: v.is_monotonic_decreasing)).all()       # a bigger loss lowers the cut-off
