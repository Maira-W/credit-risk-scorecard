"""The committed raw data and what the code makes of it."""

import json

import numpy as np
import pandas as pd
import pytest

from creditrisk import build, config, ingest, macro, models


@pytest.fixture(scope="module")
def raw():
    return build.load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return build.clean(raw)


def test_raw_files_match_the_manifest():
    assert ingest.verify() == []


def test_the_manifest_gives_a_licence_and_a_citation():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["citation"] == config.UCI_CITATION
    for name, entry in manifest["files"].items():
        assert entry["licence"] == "CC BY 4.0" and entry["source"].startswith("https://") and len(entry["sha256"]) == 64, name


def test_the_column_table_agrees_with_the_saved_uci_metadata():
    meta = json.loads((config.RAW / config.UCI_META_FILE).read_text(encoding="utf-8"))["data"]
    described = {v["name"]: v["description"] for v in meta["variables"] if v["role"] == "Feature"}
    assert described == config.UCI_NAMES
    assert meta["target_col"] == ["Y"] and meta["num_instances"] == config.UCI_ROWS
    assert "10.1016/j.eswa.2007.12.020" in config.UCI_CITATION


def test_the_raw_file_is_the_dataset_that_was_described(raw):
    assert raw.shape == (config.UCI_ROWS, 25)
    assert list(raw["id"]) == list(range(1, config.UCI_ROWS + 1))
    assert int(raw["default"].sum()) == 6636 and raw["default"].mean() == pytest.approx(0.2212)
    assert raw["limit"].min() == 10_000 and raw["limit"].max() == 1_000_000 and raw["age"].min() == 21


def test_renaming_follows_the_months():
    assert config.RENAME["X6"] == "status_sep" and config.RENAME["X11"] == "status_apr"
    assert config.RENAME["X12"] == "bill_sep" and config.RENAME["X17"] == "bill_apr"
    assert config.RENAME["X18"] == "paid_sep" and config.RENAME["X23"] == "paid_apr"
    assert len(set(config.RENAME.values())) == len(config.RENAME) == 24


def test_a_payment_settles_the_bill_of_the_month_before(raw):
    """In the first account, the August payment (689) is exactly July's bill: payments in month t pay month t-1's statement."""
    first = raw.iloc[0]
    assert first["paid_aug"] == first["bill_jul"] == 689


def test_cleaning_the_real_file_reproduces_the_committed_quality_record(cleaned):
    _, quality = cleaned
    committed = json.loads((config.PROCESSED / "data_quality.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(quality)) == committed
    assert quality["education_recoded"] == 345 and quality["marriage_recoded"] == 54 and quality["extra_copies"] == 56


def test_features_on_the_real_data_are_complete_and_in_range(cleaned):
    df = build.add_features(cleaned[0])
    assert len(df) == config.UCI_ROWS and not df[build.BEHAVIOUR].isna().any().any()
    assert df["utilisation_sep"].between(0, build.UTILISATION_CAP).all() and df["utilisation_avg"].between(0, build.UTILISATION_CAP).all()
    ratios = df[[c for c in df.columns if c.startswith("pay_ratio_")]]
    assert ratios.stack().between(0, 1).all()
    assert df["n_delinquent"].between(0, 6).all() and df["months_since_delinquent"].between(0, 6).all()
    assert ((df["n_delinquent"] == 0) == (df["months_since_delinquent"] == 6)).all()
    assert (df["n_revolving"] + df["n_paid_full"] + df["n_inactive"] + df["n_delinquent"] <= 6).all()


def test_the_split_is_reproducible_stratified_and_never_separates_twins(cleaned):
    df = cleaned[0]
    dev, hold = models.holdout_split(df)
    again = models.holdout_split(df)
    assert (dev == again[0]).all() and (hold == again[1]).all()
    assert len(dev) + len(hold) == len(df) and set(dev).isdisjoint(hold)
    assert len(hold) == pytest.approx(0.2 * len(df), rel=0.01)
    assert abs(df["default"].iloc[dev].mean() - df["default"].iloc[hold].mean()) < 0.002
    assert set(df["group"].iloc[dev]).isdisjoint(df["group"].iloc[hold])


def test_the_two_bank_loan_series_agree_where_they_overlap():
    overlap = macro.npl_overlap()
    assert len(overlap) >= 5
    assert np.allclose(overlap["value_main"], overlap["value_long"], atol=1e-5)
    series = macro.npl_ratio()
    assert series["year"].is_unique and series["year"].is_monotonic_increasing and series["year"].iloc[0] == 1998 and series["year"].iloc[-1] >= 2024
    assert series["value"].between(1, 30).all()


def test_private_credit_series_is_a_share_of_gdp():
    credit = macro.private_credit()
    assert credit["value"].between(1, 100).all() and credit["year"].is_monotonic_increasing


def test_the_policy_rate_table_is_sound():
    rate = macro.policy_rate()
    assert rate["effective_date"].is_monotonic_increasing and rate["effective_date"].is_unique
    assert (rate["policy_rate_pct"] > 0).all() and (rate["policy_rate_pct"].diff().dropna() != 0).all()      # every row is a change
    assert rate["source"].str.startswith("http").all() and rate["date_basis"].notna().all()
    assert rate.iloc[0]["effective_date"] == pd.Timestamp("2021-10-01") and rate.iloc[0]["policy_rate_pct"] == 7.25
    assert rate.loc[rate["policy_rate_pct"].idxmax(), "policy_rate_pct"] == 22.0
    for date in config.POLICY_SCENARIOS:
        assert (rate["effective_date"] == pd.Timestamp(date)).any(), date


def test_macro_context_quotes_the_series():
    c = macro.context()
    assert c["rate_peak"] == 22.0 and c["rate_low"] == 7.25 and c["npl_first_year"] == 1998 and c["npl_latest_year"] >= 2024
