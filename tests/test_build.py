"""Cleaning and features, on made-up accounts whose values are worked out by hand."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_raw

from creditrisk import build, config


def one_account(**changes) -> pd.DataFrame:
    """The account used to check every feature by hand (see the numbers in the tests below)."""
    row = {"id": 1, "limit": 100_000, "sex_code": 2, "education_code": 2, "marriage_code": 1, "age": 30, "default": 0}
    row |= dict(zip([f"status_{m}" for m in config.MONTHS], [2, 0, -1, -2, 0, 0], strict=True))
    row |= dict(zip([f"bill_{m}" for m in config.MONTHS], [50_000, 20_000, 0, -1_000, 10_000, 30_000], strict=True))
    row |= dict(zip([f"paid_{m}" for m in config.MONTHS], [5_000, 20_000, 0, 0, 0, 0], strict=True))
    row |= changes
    return pd.DataFrame([row])


def test_utilisation_features():
    f = build.add_features(one_account()).iloc[0]
    assert f["utilisation_sep"] == 0.5
    assert f["utilisation_avg"] == pytest.approx((0.5 + 0.2 + 0 + 0 + 0.1 + 0.3) / 6)      # the credit balance of -1,000 counts as 0
    assert f["utilisation_max"] == 0.5
    assert f["bill_trend"] == pytest.approx(0.5 - 0.3)


def test_utilisation_is_capped():
    f = build.add_features(one_account(bill_sep=400_000)).iloc[0]
    assert f["utilisation_sep"] == build.UTILISATION_CAP


def test_status_features():
    f = build.add_features(one_account()).iloc[0]
    assert (f["n_delinquent"], f["n_delinquent_recent"], f["months_since_delinquent"], f["worst_status"]) == (1, 1, 0, 2)
    assert (f["n_revolving"], f["n_paid_full"], f["n_inactive"]) == (3, 1, 1)
    assert f["status_trend"] == pytest.approx(2 - np.mean([0, -1, -2]))


def test_months_since_the_last_delay():
    never = build.add_features(one_account(status_sep=0)).iloc[0]
    assert never["months_since_delinquent"] == len(config.MONTHS) and never["n_delinquent"] == 0
    in_may = build.add_features(one_account(status_sep=0, status_may=3)).iloc[0]
    assert in_may["months_since_delinquent"] == 4 and in_may["n_delinquent_recent"] == 0


def test_payment_ratios():
    f = build.add_features(one_account()).iloc[0]
    # September's payment (5,000) settles August's bill (20,000); with nothing owed the ratio is 1; nothing paid is 0
    assert f["pay_ratio_sep"] == 0.25
    assert f["pay_ratio_aug"] == 1.0 and f["pay_ratio_jul"] == 1.0
    assert f["pay_ratio_jun"] == 0.0 and f["pay_ratio_may"] == 0.0
    assert f["pay_ratio_avg"] == pytest.approx((0.25 + 1 + 1 + 0 + 0) / 5)
    assert f["pay_ratio_total"] == pytest.approx(5_000 / 60_000)          # 20,000 + 0 + 0 + 10,000 + 30,000 was owed
    assert f["paid_total"] == 25_000


def test_overpaying_counts_as_paying_in_full():
    f = build.add_features(one_account(paid_sep=90_000)).iloc[0]
    assert f["pay_ratio_sep"] == 1.0


def test_features_listed_in_build_exist():
    features = build.add_features(build.clean(make_raw())[0])
    missing = [c for c in build.BEHAVIOUR if c not in features.columns]
    assert missing == []
    assert not features[build.ENGINEERED].isna().any().any()


def test_undocumented_codes_are_put_with_others():
    raw = make_raw()
    raw["education_code"] = np.resize([0, 1, 2, 3, 4, 5, 6], len(raw))
    raw["marriage_code"] = np.resize([0, 1, 2, 3], len(raw))
    clean, quality = build.clean(raw)
    assert set(clean["education_code"]) == {1, 2, 3, 4} and set(clean["marriage_code"]) == {1, 2, 3}
    assert quality["education_recoded"] == int(raw["education_code"].isin([0, 5, 6]).sum())
    assert quality["marriage_recoded"] == int((raw["marriage_code"] == 0).sum())
    assert set(clean["education"]) == set(config.EDUCATION_LABELS.values())
    assert clean.loc[raw["education_code"] == 5, "education"].eq("Other or unknown").all()


def test_age_bands_cover_everyone_without_gaps():
    raw = make_raw()
    raw["age"] = np.resize([21, 29, 30, 39, 40, 49, 50, 79], len(raw))
    clean, _ = build.clean(raw)
    assert clean.groupby("age")["age_band"].nunique().eq(1).all()
    assert dict(zip(clean["age"], clean["age_band"], strict=True)) == {21: "21-29", 29: "21-29", 30: "30-39", 39: "30-39", 40: "40-49",
                                                                       49: "40-49", 50: "50 and over", 79: "50 and over"}


def test_identical_accounts_share_a_group_whatever_their_outcome():
    raw = make_raw(50)
    raw.loc[1, raw.columns.difference(["id", "default"])] = raw.loc[0, raw.columns.difference(["id", "default"])].to_numpy()
    raw.loc[1, "default"] = 1 - raw.loc[0, "default"]
    clean, quality = build.clean(raw)
    assert clean.loc[0, "group"] == clean.loc[1, "group"]
    assert clean["group"].nunique() == 49 and quality["rows_in_repeated_groups"] == 2 and quality["extra_copies"] == 1


def test_quality_record_counts_what_it_says():
    raw = make_raw()
    _, q = build.clean(raw)
    assert q["rows"] == len(raw) and q["defaults"] == int(raw["default"].sum())
    assert q["accounts_with_a_negative_bill"] == int((raw[config.BILL] < 0).any(axis=1).sum())
    assert q["over_limit_in_september"] == int((raw["bill_sep"] > raw["limit"]).sum())
    assert q["late_share_by_month"]["sep"] == pytest.approx((raw["status_sep"] >= 1).mean())


@pytest.mark.parametrize("damage", [
    lambda d: d.__setitem__("limit", d["limit"].where(d["id"] != 3)),                       # a missing value
    lambda d: d.__setitem__("id", d["id"].where(d["id"] != 3, 2)),                          # a repeated id
    lambda d: d.__setitem__("default", d["default"].where(d["id"] != 3, 2)),                # an outcome that is neither 0 nor 1
    lambda d: d.__setitem__("sex_code", d["sex_code"].where(d["id"] != 3, 3)),
    lambda d: d.__setitem__("status_sep", d["status_sep"].where(d["id"] != 3, 12)),
    lambda d: d.__setitem__("paid_sep", d["paid_sep"].where(d["id"] != 3, -5)),
    lambda d: d.__setitem__("limit", d["limit"].where(d["id"] != 3, 0)),
    lambda d: d.__setitem__("age", d["age"].where(d["id"] != 3, 5)),
])
def test_a_damaged_table_is_refused(damage):
    raw = make_raw()
    damage(raw)
    with pytest.raises(ValueError):
        build.validate(raw)


def test_a_table_with_the_wrong_columns_is_refused(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"ID": [1], "X1": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        build.load_raw(path)


def test_the_valid_table_passes():
    build.validate(make_raw())
