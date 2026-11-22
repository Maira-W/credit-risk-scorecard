"""The split, the baseline, the challenger and cross-validation, on made-up accounts."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import config, models, validation


@pytest.fixture(scope="module")
def accounts():
    """4,000 made-up accounts, 100 of them twins (same group), whose default depends on `signal` and not on `noise`."""
    rng = np.random.default_rng(5)
    n = 4000
    signal, noise = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.3 + 1.2 * signal)))
    frame = pd.DataFrame({"signal": signal, "noise": noise, "status_sep": np.where(signal > 0.5, 2, np.where(signal > -0.3, 0, -1)),
                          "default": (rng.uniform(size=n) < p).astype(int), "group": np.arange(n)})
    frame.loc[3900:3999, "group"] = np.repeat(np.arange(50), 2) + 10_000        # 50 pairs of twins at the end
    return frame, p


def test_the_holdout_split_is_stratified_disjoint_and_never_separates_twins(accounts):
    frame, _ = accounts
    dev, hold = models.holdout_split(frame)
    assert set(dev).isdisjoint(hold) and len(dev) + len(hold) == len(frame)
    assert len(hold) == pytest.approx(0.2 * len(frame), rel=0.03)
    assert abs(frame["default"].iloc[dev].mean() - frame["default"].iloc[hold].mean()) < 0.01
    assert set(frame["group"].iloc[dev]).isdisjoint(frame["group"].iloc[hold])


def test_the_split_depends_on_the_seed_and_repeats_for_the_same_seed(accounts):
    frame, _ = accounts
    a, b = models.holdout_split(frame, seed=1), models.holdout_split(frame, seed=1)
    c = models.holdout_split(frame, seed=2)
    assert (a[1] == b[1]).all() and not (len(a[1]) == len(c[1]) and (np.sort(a[1]) == np.sort(c[1])).all())


def test_cross_validation_folds_use_every_account_once_per_repeat_and_keep_twins_together(accounts):
    frame, _ = accounts
    y, groups = frame["default"].to_numpy(), frame["group"].to_numpy()
    seen = {}
    for repeat, _fold, train, valid in models.cv_splits(y, groups, folds=4, repeats=2):
        assert set(train).isdisjoint(valid) and set(groups[train]).isdisjoint(groups[valid])
        seen.setdefault(repeat, []).extend(valid)
    assert all(sorted(v) == list(range(len(frame))) for v in seen.values()) and len(seen) == 2


def test_the_naive_rule_gives_each_status_its_own_default_rate():
    X = pd.DataFrame({"status_sep": [0, 0, 1, 1, 1, 2]})
    model = models.RateByStatus().fit(X, [0, 1, 1, 1, 0, 1])
    p = model.predict_proba(pd.DataFrame({"status_sep": [0, 1, 2, 9]}))
    assert list(p) == pytest.approx([0.5, 2 / 3, 1.0, 4 / 6])                        # an unseen status gets the overall rate


def test_the_challenger_ranks_by_the_signal_and_its_contributions_add_up(accounts):
    frame, _ = accounts
    train, test = frame.iloc[:3000], frame.iloc[3000:]
    model = models.Challenger(["signal", "noise"], {"n_estimators": 100}).fit(train, train["default"])
    p = model.predict_proba(test)
    assert validation.auc(test["default"], p) > 0.72
    shap = model.contributions(test)
    assert shap.abs().mean()["signal"] > 3 * shap.abs().mean()["noise"]
    raw = model.model_.predict(test[["signal", "noise"]], pred_contrib=True)
    assert 1 / (1 + np.exp(-raw.sum(axis=1))) == pytest.approx(p, abs=1e-6)       # SHAP values plus the base value are the model's log-odds


def test_the_challenger_is_deterministic(accounts):
    frame, _ = accounts
    a = models.Challenger(["signal", "noise"], {"n_estimators": 50}).fit(frame, frame["default"]).predict_proba(frame)
    b = models.Challenger(["signal", "noise"], {"n_estimators": 50}).fit(frame, frame["default"]).predict_proba(frame)
    assert (a == b).all()


def test_recalibration_repairs_an_overconfident_model(accounts):
    frame, p_true = accounts
    y = frame["default"].to_numpy()
    z = np.log(p_true / (1 - p_true))
    overconfident = 1 / (1 + np.exp(-2 * z))
    assert validation.calibration_fit(y, overconfident)[1] == pytest.approx(0.5, abs=0.1)
    platt = models.Platt().fit(overconfident, y)
    assert validation.calibration_fit(y, platt.predict(overconfident))[1] == pytest.approx(1.0, abs=0.05)
    assert validation.auc(y, platt.predict(overconfident)) == pytest.approx(validation.auc(y, overconfident), abs=1e-6)     # the ranking is untouched


def test_out_of_fold_predictions_cover_every_account(accounts):
    frame, _ = accounts
    p = models.out_of_fold(lambda: models.RateByStatus(), frame, frame["default"], frame["group"], folds=4)
    assert not np.isnan(p).any() and ((p >= 0) & (p <= 1)).all() and validation.auc(frame["default"], p) > 0.6


def test_cross_validate_gives_a_row_per_model_repeat_and_fold(accounts):
    frame, _ = accounts
    table = models.cross_validate({"naive": lambda: models.RateByStatus(), "boosting": lambda: models.Challenger(["signal"], {"n_estimators": 30})},
                                  frame, frame["default"], frame["group"], folds=3, repeats=2)
    assert len(table) == 2 * 2 * 3 and set(table["model"]) == {"naive", "boosting"}
    assert set(table.columns) >= {"auc", "ks", "brier", "log_loss", "gini"}
    assert table.groupby("model")["auc"].mean()["boosting"] > table.groupby("model")["auc"].mean()["naive"] - 0.05


def test_tuning_returns_the_trial_with_the_lowest_log_loss(accounts):
    frame, _ = accounts
    best, table = models.tune_challenger(frame.iloc[:2000], frame["default"].iloc[:2000], frame["group"].iloc[:2000], ["signal", "noise"], trials=3, folds=3)
    assert len(table) == 3 and set(best) == set(models.SEARCH_SPACE)
    row = table.sort_values(["cv_log_loss", "trial"]).iloc[0]
    assert all(row[k] == pytest.approx(v) for k, v in best.items())
    assert {k: table.iloc[0][k] for k in config.CHALLENGER_PARAMS} == config.CHALLENGER_PARAMS      # the default settings are always trial 0
