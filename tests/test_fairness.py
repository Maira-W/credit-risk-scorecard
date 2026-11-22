"""The fairness audit, on made-up groups with known approval and error rates."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import config, fairness


@pytest.fixture
def small_groups(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_SIZE", 1)
    # group A: 10 accounts, 8 approved (p <= 0.1), 2 declined; group B: 10 accounts, 4 approved
    return pd.DataFrame({
        "grp": ["A"] * 10 + ["B"] * 10,
        "probability": [0.05] * 8 + [0.4] * 2 + [0.05] * 4 + [0.4] * 6,
        "default": [0] * 8 + [1, 1] + [0] * 4 + [0, 0, 0, 1, 1, 1],
    })


def test_group_table_by_hand(small_groups):
    t = fairness.group_table(small_groups, "grp", cutoff=0.1).set_index("group")
    assert t.loc["A", "approval_rate"] == 0.8 and t.loc["B", "approval_rate"] == 0.4
    assert t.loc["B", "disparate_impact"] == pytest.approx(0.5) and t.loc["A", "disparate_impact"] == 1.0
    assert bool(t.loc["B", "flagged"]) and not bool(t.loc["A", "flagged"])
    assert t.loc["A", "defaulters_declined"] == 1.0 and t.loc["B", "defaulters_declined"] == 1.0
    assert t.loc["A", "good_declined"] == 0.0 and t.loc["B", "good_declined"] == pytest.approx(3 / 7)
    assert t.loc["A", "default_rate"] == pytest.approx(0.2) and t.loc["B", "default_rate"] == pytest.approx(0.3)
    assert t.loc["A", "calibration_ratio"] == pytest.approx((0.05 * 8 + 0.4 * 2) / 10 / 0.2)
    assert t["approval_rate_overall"].iloc[0] == pytest.approx(0.6)
    assert t["share"].sum() == pytest.approx(1.0)


def test_a_gap_just_inside_the_four_fifths_rule_is_not_flagged(small_groups):
    frame = small_groups.copy()
    frame.loc[frame["grp"] == "B", "probability"] = [0.05] * 7 + [0.4] * 3            # B approved at 0.7, A at 0.8: ratio 0.875
    t = fairness.group_table(frame, "grp", cutoff=0.1).set_index("group")
    assert t.loc["B", "disparate_impact"] == pytest.approx(0.875) and not bool(t.loc["B", "flagged"])


def test_small_groups_are_shown_but_not_judged(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_SIZE", 15)
    frame = pd.DataFrame({"g": ["big"] * 30 + ["tiny"] * 3, "probability": [0.05] * 30 + [0.9] * 3, "default": [0] * 30 + [1] * 3})
    t = fairness.group_table(frame, "g", cutoff=0.1).set_index("group")
    assert np.isnan(t.loc["tiny", "disparate_impact"]) and not bool(t.loc["tiny", "flagged"])
    assert t.loc["tiny", "accounts"] == 3


def test_audit_stacks_the_attributes(small_groups):
    frame = small_groups.assign(other=["x", "y"] * 10)
    t = fairness.audit(frame, ["grp", "other"], cutoff=0.1)
    assert set(t["attribute"]) == {"grp", "other"} and len(t) == 4


def test_auc_inside_a_group_needs_both_outcomes(small_groups):
    t = fairness.group_table(small_groups.assign(default=[0] * 10 + [0, 1] * 5), "grp", cutoff=0.1).set_index("group")
    assert np.isnan(t.loc["A", "auc"]) and not np.isnan(t.loc["B", "auc"])


def test_impact_interval_brackets_the_ratio_and_narrows_with_more_accounts(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_SIZE", 50)
    rng = np.random.default_rng(3)

    def accounts(n):
        group = rng.choice(["A", "B"], n)
        probability = np.where(group == "A", rng.uniform(0, 0.2, n), rng.uniform(0.05, 0.3, n))     # B is approved less often at a cut-off of 0.1
        return pd.DataFrame({"grp": group, "probability": probability, "default": 0})

    small, large = accounts(400), accounts(8000)
    narrow = fairness.impact_interval(large, "grp", cutoff=0.1, draws=300)
    wide = fairness.impact_interval(small, "grp", cutoff=0.1, draws=300)
    point = fairness.group_table(large, "grp", 0.1).set_index("group")["disparate_impact"]
    b = narrow.set_index("group").loc["B"]
    assert b["di_low"] < point["B"] < b["di_high"] and b["di_high"] < 0.8          # B is under four-fifths of A's rate, and clearly so
    assert (wide.set_index("group").loc["B", "di_high"] - wide.set_index("group").loc["B", "di_low"]) > (b["di_high"] - b["di_low"])
    assert narrow.set_index("group").loc["A", "di_high"] == 1.0                    # the best-treated group is the reference in nearly every resample


def test_groups_that_are_not_judged_get_no_range(monkeypatch):
    monkeypatch.setattr(config, "MIN_GROUP_SIZE", 15)
    frame = pd.DataFrame({"g": ["big"] * 60 + ["tiny"] * 3, "probability": [0.05, 0.4] * 30 + [0.9] * 3, "default": [0, 1] * 30 + [1] * 3})
    t = fairness.audit(frame, ["g"], cutoff=0.1, draws=50).set_index("group")
    assert np.isnan(t.loc["tiny", "di_low"]) and not np.isnan(t.loc["big", "di_low"])


def test_proxy_strength_is_high_for_a_stand_in_and_about_half_for_unrelated_inputs():
    rng = np.random.default_rng(1)
    n = 3000
    target = rng.integers(0, 2, n)
    X = pd.DataFrame({"stand_in": target + rng.normal(scale=0.5, size=n), "noise": rng.normal(size=n), "noise2": rng.normal(size=n)})
    assert fairness.proxy_strength(X, target, ["stand_in", "noise"]) > 0.85
    assert fairness.proxy_strength(X, target, ["noise", "noise2"]) < 0.58
