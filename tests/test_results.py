"""The committed results: consistent with each other, and the ones a reader is shown."""

import json

import pandas as pd
import pytest
from results_checks import check_results

from creditrisk import config


@pytest.fixture(scope="module")
def summary():
    return check_results(config.PROCESSED)


def test_the_committed_results_are_internally_consistent(summary):
    assert summary["data"]["accounts"] == 30_000 and summary["settings"]["sample"] is None


def test_they_come_from_the_full_settings(summary):
    s = summary["settings"]
    assert (s["cv_folds"], s["cv_repeats"], s["tuning_trials"], s["bootstrap_draws"]) == (config.CV_FOLDS, config.CV_REPEATS, config.TUNING_TRIALS, config.BOOTSTRAP_DRAWS)


def test_the_split_is_80_20_and_stratified(summary):
    d = summary["data"]
    assert (d["development"], d["holdout"]) == (24_000, 6_000)
    assert d["default_rate_development"] == pytest.approx(d["default_rate_holdout"], abs=0.002) and d["default_rate"] == pytest.approx(0.2212, abs=1e-4)


def test_the_economics_use_the_documented_assumptions(summary):
    e, roll = summary["economics"], summary["rollrates"]
    assert (e["apr"], e["opex"]) == (config.ECON_APR, config.ECON_OPEX)
    assert (roll["write_off_status"], roll["horizon"], roll["lgd"], roll["collection_cost"]) == (
        config.ECON_WRITEOFF_STATUS, config.ECON_HORIZON, config.ECON_LGD, config.ECON_COLLECTION_COST)
    assert e["loss"] == pytest.approx(config.ECON_LGD * roll["base_probability"] + config.ECON_COLLECTION_COST)      # measured from roll rates, not guessed
    assert roll["loss_stress"] == config.ECON_LOSS_STRESS
    policy = pd.read_csv(config.POLICY_RATE_FILE, parse_dates=["effective_date"])
    assert e["funding"] == pytest.approx(policy.iloc[-1]["policy_rate_pct"] / 100) and e["policy_rate_date"] == str(policy.iloc[-1]["effective_date"].date())
    assert 0.8 < e["revolve_share"] < 0.95


def test_the_roll_rates_say_what_the_documents_claim(summary):
    split = pd.read_csv(config.PROCESSED / "rollrates_split.csv")
    base = split[split["write_off_status"] == config.ECON_WRITEOFF_STATUS].set_index("group")
    assert base.loc["Late in September", "loss"] > base.loc["Current in September", "loss"]            # an account that is already late costs more than a first missed payment
    r = summary["rollrates"]
    assert r["p_cure_from_first_late"] > 0.2 and r["p_stay_first_late"] > r["p_worse_from_first_late"]
    assert base.loc["Current in September", "loss"] <= summary["economics"]["loss"] <= base.loc["Late in September", "loss"]      # the base case is the mix of the two


def test_the_scorecard_is_a_readable_size_and_leaves_out_the_protected_attributes(summary):
    sc = summary["scorecard"]
    assert 5 <= sc["n_features"] <= 12 and sc["n_bins"] <= 6 * sc["n_features"]
    assert not set(sc["features"]) & set(config.DEMOGRAPHICS)
    assert not set(sc["features"]) & {c[0] for c in config.MONTH_COUNT_CANDIDATES}


def test_the_headline_claims_hold(summary):
    m = summary["metrics"]
    assert m["scorecard"]["auc"] > m["naive"]["auc"] + 0.04                         # the scorecard is clearly better than last month's status alone
    assert m["challenger"]["auc"] > m["scorecard"]["auc"]
    assert abs(m["scorecard"]["calibration_slope"] - 1) < 0.1 and abs(m["challenger"]["calibration_slope"] - 1) < 0.1
    assert summary["psi"]["score"] < 0.1


def test_the_elimination_loop_ran_on_the_real_data():
    screening = pd.read_csv(config.PROCESSED / "scorecard_screening.csv")
    reasons = screening["reason"].fillna("")
    assert reasons.str.startswith("correlated").any() and reasons.str.startswith(("p-value", "coefficient")).any() and reasons.str.startswith("information value").any()


def test_the_summary_is_strict_json():
    text = (config.PROCESSED / "summary.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in summary.json"))
