"""Weight-of-evidence binning, on made-up data whose answer is known."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import binning


def test_woe_and_iv_by_hand():
    woe, iv = binning.woe_iv([10, 90], [90, 10], smoothing=0)
    # share of bad in bin 1 is 0.1, share of good 0.9: WoE = ln(0.1 / 0.9)
    assert woe[0] == pytest.approx(np.log(0.1 / 0.9))
    assert woe[1] == pytest.approx(np.log(0.9 / 0.1))
    assert iv.sum() == pytest.approx(2 * (0.9 - 0.1) * np.log(9))


def test_a_bin_with_no_bad_accounts_still_has_a_finite_weight():
    woe, iv = binning.woe_iv([0, 50], [50, 50])
    assert np.isfinite(woe).all() and np.isfinite(iv).all()


def test_information_value_is_zero_when_bins_do_not_separate():
    assert binning.information_value([20, 20, 20], [80, 80, 80]) == pytest.approx(0, abs=1e-3)


def test_similar_neighbours_merge_first():
    groups = binning.merge_neighbours([10, 10, 50], [90, 90, 50], min_share=0.0, max_bins=6, iv_tolerance=0.01)
    assert groups == [[0, 1], [2]]


def test_too_small_bin_joins_the_neighbour_with_the_closer_rate():
    # the middle bin (10 accounts of 400) is under 5% and has a rate nearer the first bin's
    groups = binning.merge_neighbours([20, 1, 60], [180, 9, 130], min_share=0.05, max_bins=6, iv_tolerance=0.0)
    assert groups == [[0, 1], [2]]


def test_monotone_rule_removes_reversals():
    bad = np.array([10, 30, 20, 50.0])
    good = np.array([90, 70, 80, 50.0])
    groups = binning.merge_neighbours(bad, good, min_share=0.0, max_bins=6, iv_tolerance=0.0, monotone="up")
    rates = [bad[g].sum() / (bad[g].sum() + good[g].sum()) for g in groups]
    assert rates == sorted(rates)
    assert len(groups) < 4


def test_max_bins_is_respected():
    rng = np.random.default_rng(1)
    bad, good = rng.integers(20, 60, 12), rng.integers(100, 200, 12)
    assert len(binning.merge_neighbours(bad, good, min_share=0.0, max_bins=4, iv_tolerance=0.0)) == 4


def test_numeric_binning_finds_the_planted_breaks():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, 30_000)
    rate = np.where(x < 3, 0.10, np.where(x < 7, 0.30, 0.60))
    y = (rng.uniform(size=len(x)) < rate).astype(int)
    b = binning.fit_binning(x, y, "x", "numeric", "up", iv_tolerance=0.005)     # a split that adds under 0.005 is noise here
    assert b.n_bins == 3
    assert b.edges == pytest.approx([3, 7], abs=0.5)
    assert list(b.table["woe"]) == sorted(b.table["woe"])
    assert b.iv > 0.3


def test_a_value_on_a_cut_point_goes_to_the_bin_below():
    b = binning.FeatureBinning(name="x", kind="numeric", edges=np.array([2.0, 5.0]),
                               table=pd.DataFrame({"label": ["a", "b", "c"], "woe": [-1.0, 0.0, 1.0]}))
    assert list(b.bin_index([1.9, 2.0, 2.1, 5.0, 5.1])) == [0, 0, 1, 1, 2]


def test_categorical_binning_merges_categories_of_similar_risk():
    rng = np.random.default_rng(4)
    x = rng.choice(list("abcd"), 20_000)
    rate = pd.Series(x).map({"a": 0.10, "b": 0.11, "c": 0.40, "d": 0.41}).to_numpy()
    y = (rng.uniform(size=len(x)) < rate).astype(int)
    b = binning.fit_binning(x, y, "x", "categorical")
    assert sorted(sorted(g) for g in b.groups) == [["a", "b"], ["c", "d"]]
    assert b.woe(["a", "d"])[0] < 0 < b.woe(["a", "d"])[1]


def test_unseen_category_gets_the_average_weight():
    b = binning.fit_binning(["a"] * 60 + ["b"] * 60, [0] * 50 + [1] * 10 + [0] * 20 + [1] * 40, "x", "categorical")
    assert b.woe(["zzz"])[0] == 0.0
    assert b.bin_index(["zzz"])[0] == -1


def test_missing_values_get_their_own_bin():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 6000)
    y = (rng.uniform(size=6000) < 0.2).astype(int)
    x[:600] = np.nan
    y[:600] = (rng.uniform(size=600) < 0.7).astype(int)
    b = binning.fit_binning(x, y, "x", "numeric")
    assert b.missing_bin is not None
    assert b.table.loc[b.missing_bin, "label"] == "missing"
    assert b.table.loc[b.missing_bin, "n"] == 600
    assert b.woe([np.nan])[0] == b.table.loc[b.missing_bin, "woe"] > 0


def test_labels_of_whole_numbers():
    assert binning.numeric_labels(np.array([1.5, 3.5]), True, 0, 9) == ["1 or less", "2 to 3", "4 or more"]
    assert binning.numeric_labels(np.array([0.5, 1.5]), True, 0, 2) == ["0", "1", "2"]
    assert binning.numeric_labels(np.array([45_000.0]), True, 10_000, 1_000_000) == ["45,000 or less", "45,001 or more"]


def test_labels_of_other_numbers():
    assert binning.numeric_labels(np.array([0.2, 0.5]), False) == ["up to 0.2", "over 0.2 up to 0.5", "over 0.5"]
    assert binning.numeric_labels(np.array([]), False) == ["all values"]


def test_cut_points_are_rounded_to_the_spread_of_the_data():
    values = np.random.default_rng(2).uniform(10_000, 410_000, 5000)
    cuts = binning.round_cuts(np.array([44_873.0, 127_411.0]), values)
    assert list(cuts) == [45_000.0, 127_000.0]


def test_a_feature_that_never_varies_has_one_bin():
    b = binning.fit_binning(np.ones(100), np.r_[np.zeros(80), np.ones(20)], "x")
    assert b.n_bins == 1 and b.iv == pytest.approx(0)


@pytest.mark.parametrize("y", [np.zeros(50), np.ones(50)])
def test_both_outcomes_are_required(y):
    with pytest.raises(ValueError):
        binning.fit_binning(np.arange(50), y)


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        binning.fit_binning([1, 2], [0, 1], kind="ordinal")
