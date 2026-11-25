"""The analysis: split the accounts, build and validate the models, audit them for fairness, and price the cut-off.

Everything is chosen on the 80% development set (bins, features, challenger settings, recalibration); the 20% hold-out is used once,
to report. Results are written to data/processed as CSV files and summary.json; the figures, dashboard and documents are made from
those files alone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import advanced, build, config, economics, fairness, macro, models, rollrates, scorecard, validation

MODEL_NAMES = {
    "naive": "Naive: last month's status only",
    "scorecard": "Scorecard",
    "scorecard_demographics": "Scorecard + sex, age, education, marriage",
    "challenger": "Gradient boosting",
    "challenger_demographics": "Gradient boosting + sex, age, education, marriage",
}
DECISION_MODELS = ("naive", "scorecard", "challenger")
DEMOGRAPHIC_CANDIDATES = (("sex_code", "categorical", None), ("education_code", "categorical", None),
                          ("marriage_code", "categorical", None), ("age", "numeric", None))
STATUS_NAMES = {-2: "No spending", -1: "Paid in full", 0: "Paid the minimum", 1: "1 month late", 2: "2 months late"}
COMPARISONS = (("scorecard", "naive"), ("challenger", "scorecard"), ("scorecard_demographics", "scorecard"),
               ("challenger_demographics", "challenger"))
PROBABILITY_DECIMALS = 3          # the challenger's probabilities are rounded to 0.1 of a percentage point, so that the cut-off tables stay small
RISK_BANDS = ((0.0, 0.10, "Very low risk (under 10%)"), (0.10, 0.15, "Low (10% to 15%)"), (0.15, 0.25, "Medium (15% to 25%)"),
              (0.25, 0.50, "High (25% to 50%)"), (0.50, 1.01, "Very high (50% and over)"))


@dataclass(frozen=True)
class Settings:
    cv_folds: int = config.CV_FOLDS
    cv_repeats: int = config.CV_REPEATS
    tuning_trials: int = config.TUNING_TRIALS
    bootstrap_draws: int = config.BOOTSTRAP_DRAWS
    sample: int | None = None            # use only this many accounts: for the tests, which run the whole analysis on a small sample


FAST = Settings(cv_folds=3, cv_repeats=1, tuning_trials=2, bootstrap_draws=60, sample=5000)


def factories(params: dict) -> dict:
    return {
        "naive": lambda: models.RateByStatus(),
        "scorecard": lambda: scorecard.Scorecard(),
        "scorecard_demographics": lambda: scorecard.Scorecard((*config.SCORECARD_CANDIDATES, *DEMOGRAPHIC_CANDIDATES)),
        "challenger": lambda: models.Challenger(build.BEHAVIOUR, params),
        "challenger_demographics": lambda: models.Challenger([*build.BEHAVIOUR, *config.DEMOGRAPHICS], params),
    }


# ---------------------------------------------------------------------------
# Who defaults: descriptive tables
# ---------------------------------------------------------------------------
def wilson(defaults, accounts, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a default rate."""
    p = defaults / accounts
    centre = (p + z * z / (2 * accounts)) / (1 + z * z / accounts)
    half = z * math.sqrt(p * (1 - p) / accounts + z * z / (4 * accounts * accounts)) / (1 + z * z / accounts)
    return centre - half, centre + half


def _bands(series: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True).astype(str)


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Default rate of the accounts in each segment, one block of segments per variable."""
    late = df["status_sep"].clip(upper=3).map({**STATUS_NAMES, 3: "3 or more months late"})
    credit_used = pd.cut(df["bill_sep"] / df["limit"], [-np.inf, 0, 0.2, 0.5, 0.8, 1.0, np.inf],
                         labels=["None or in credit", "Up to 20%", "20% to 50%", "50% to 80%", "80% to 100%", "Over the limit"])
    paid = pd.cut(df["pay_ratio_sep"], [-np.inf, 0, 0.1, 0.5, 0.999999, np.inf],
                  labels=["Nothing", "Under 10%", "10% to 50%", "50% to under all", "All (or nothing was owed)"])
    limit = pd.cut(df["limit"], [0, 50_000, 100_000, 200_000, 300_000, np.inf],
                   labels=["NT$50,000 or less", "NT$50,000 to 100,000", "NT$100,000 to 200,000", "NT$200,000 to 300,000", "Over NT$300,000"])
    months = df["n_delinquent"].clip(upper=4).map({0: "0 months", 1: "1 month", 2: "2 months", 3: "3 months", 4: "4 or more months"})
    blocks = [("Repayment status last month (September)", late, list(STATUS_NAMES.values()) + ["3 or more months late"]),
              ("Months late in the last six", months, ["0 months", "1 month", "2 months", "3 months", "4 or more months"]),
              ("Share of the previous bill paid in September", paid.astype(str), list(paid.cat.categories)),
              ("Balance as a share of the credit limit", credit_used.astype(str), list(credit_used.cat.categories)),
              ("Credit limit", limit.astype(str), list(limit.cat.categories)),
              ("Sex", df["sex"], ["Female", "Male"]),
              ("Age", df["age_band"], [b[2] for b in config.AGE_BANDS]),
              ("Education", df["education"], list(config.EDUCATION_LABELS.values())),
              ("Marital status", df["marriage"], list(config.MARRIAGE_LABELS.values()))]
    rows = []
    for variable, values, order in blocks:
        frame = pd.DataFrame({"group": values.astype(str), "default": df["default"]})
        for position, group in enumerate(order):
            part = frame[frame["group"] == str(group)]
            if len(part) == 0:
                continue
            low, high = wilson(part["default"].sum(), len(part))
            rows.append({"variable": variable, "position": position, "group": group, "accounts": len(part), "share": len(part) / len(df),
                         "defaults": int(part["default"].sum()), "default_rate": part["default"].mean(), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fitting and validating
# ---------------------------------------------------------------------------
def holdout_table(y, preds: dict, draws: int, comparisons=COMPARISONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold-out metrics of every model with bootstrap intervals, and the paired differences between models."""
    boots = {name: validation.bootstrap(func, y, preds, draws) for name, func in
             (("auc", validation.auc), ("ks", validation.ks), ("brier", validation.brier))}
    rows = []
    for name, p in preds.items():
        row = {"model": name, **{m: f(y, p) for m, f in validation.METRICS.items()}}
        for metric, draws_table in boots.items():
            row[f"{metric}_low"], row[f"{metric}_high"] = validation.interval(draws_table[name])
        row["gini_low"], row["gini_high"] = 2 * row["auc_low"] - 1, 2 * row["auc_high"] - 1
        row["calibration_intercept"], row["calibration_slope"] = validation.calibration_fit(y, p)
        lift = validation.lift_table(y, p)
        row["capture_top_decile"], row["capture_top_3_deciles"] = float(lift["capture"].iloc[0]), float(lift["capture"].iloc[2])
        rows.append(row)
    pairs = []
    for a, b in comparisons:
        for metric, draws_table in boots.items():
            diff = draws_table[a] - draws_table[b]
            low, high = validation.interval(diff)
            measure = validation.METRICS[metric]
            pairs.append({"model": a, "against": b, "metric": metric, "difference": float(measure(y, preds[a]) - measure(y, preds[b])),
                          "low": low, "high": high, "share_positive": float((diff > 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def subset_metrics(hold: pd.DataFrame, y, preds: dict) -> pd.DataFrame:
    """How well each model ranks within the accounts that were not late last month, and within those that were. The first group is
    the hard one: a rule that only looks at last month's status has nothing to go on there."""
    rows = []
    for label, mask in (("Not late in September", (hold["status_sep"] <= 0).to_numpy()), ("Late in September", (hold["status_sep"] >= 1).to_numpy())):
        for name in DECISION_MODELS:
            p = preds[name][mask]
            lift = validation.lift_table(y[mask], p)
            rows.append({"subset": label, "model": name, "accounts": int(mask.sum()), "defaults": int(y[mask].sum()), "default_rate": float(y[mask].mean()),
                         "auc": validation.auc(y[mask], p), "ks": validation.ks(y[mask], p), "capture_top_decile": float(lift["capture"].iloc[0])})
    return pd.DataFrame(rows)


def example_accounts(card: scorecard.Scorecard, hold: pd.DataFrame, y, p) -> list[dict]:
    """Three hold-out accounts, from the risky end, the middle and the safe end of the scorecard, with the reasons it gave."""
    score = card.score(hold)
    order = np.argsort(score, kind="stable")
    out = []
    for label, quantile in (("Riskier than 90% of accounts", 0.10), ("In the middle", 0.50), ("Safer than 90% of accounts", 0.90)):
        i = int(order[int(quantile * (len(order) - 1))])
        reasons = card.reason_codes(hold.iloc[[i]], k=3)[0]
        out.append({"label": label, "score": int(score[i]), "probability": float(p[i]), "defaulted": int(y[i]),
                    "status_sep": int(hold.iloc[i]["status_sep"]), "limit": int(hold.iloc[i]["limit"]), "bill_sep": int(hold.iloc[i]["bill_sep"]),
                    "reasons": [{"feature": config.feature_name(f), "bin": b, "points_lost": int(lost)} for f, b, lost in reasons]})
    return out


def psi_table(card: scorecard.Scorecard, dev: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    rows = [{"item": "Score", "psi": validation.psi(card.score(dev), card.score(hold))}]
    rows += [{"item": name, "psi": validation.psi(dev[name], hold[name])} for name in card.features_]
    return pd.DataFrame(rows)


def risk_bands(y, p, balance) -> pd.DataFrame:
    rows = []
    for low, high, label in RISK_BANDS:
        mask = (p >= low) & (p < high)
        if mask.sum() == 0:
            continue
        rows.append({"band": label, "accounts": int(mask.sum()), "share": float(mask.mean()), "default_rate": float(y[mask].mean()),
                     "share_of_defaults": float(y[mask].sum() / y.sum()), "mean_probability": float(p[mask].mean()),
                     "mean_balance": float(balance[mask].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def scorecard_variants(dev: pd.DataFrame, y_dev, folds: int) -> pd.DataFrame:
    """Cross-validated AUC and size of the scorecard for each limit on the correlation between features, and with the month counts
    that were left out of the candidates. This is how the settings in config.py were chosen."""
    rows = []

    def trial(label: str, candidates, **settings_) -> None:
        aucs, counts = [], []
        for _, _, train, valid in models.cv_splits(y_dev, dev["group"].to_numpy(), folds, 1):
            card = scorecard.Scorecard(candidates, **settings_).fit(dev.iloc[train], y_dev[train])
            aucs.append(validation.auc(y_dev[valid], card.predict_proba(dev.iloc[valid])))
            counts.append(len(card.features_))
        rows.append({"variant": label, "cv_auc": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs)), "features": float(np.mean(counts))})

    for limit in config.CORRELATION_CHECKS:
        trial(f"Correlation limit {limit:.1f}" + (" (used)" if limit == config.MAX_CORRELATION else ""), config.SCORECARD_CANDIDATES, max_correlation=limit)
    trial(f"With the month counts, correlation limit {config.MAX_CORRELATION:.1f}", (*config.SCORECARD_CANDIDATES, *config.MONTH_COUNT_CANDIDATES))
    return pd.DataFrame(rows)


def strategy_profits(y, balance, approvals: dict, econ: economics.Economics, draws: int, seed: int = config.SEED) -> pd.DataFrame:
    """Average profit per account (approved or not) of each approval rule, with a bootstrap interval, and the paired difference
    of every rule from approving everyone."""
    realised = balance * np.where(y == 1, -econ.loss, econ.margin)
    names = list(approvals)
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {name: np.empty(draws) for name in names}
    for d in range(draws):
        pick = rng.integers(0, n, n)
        for name in names:
            out[name][d] = (realised[pick] * approvals[name][pick]).mean()
    rows = []
    for name in names:
        low, high = validation.interval(pd.Series(out[name]))
        gap = pd.Series(out[name] - out["approve_all"])
        gap_low, gap_high = validation.interval(gap)
        against = pd.Series(out[name] - out["scorecard"])
        against_low, against_high = validation.interval(against)
        approved = approvals[name]
        rows.append({"rule": name, "approved_share": float(approved.mean()),
                     "default_rate_approved": float(y[approved].mean()) if approved.any() else float("nan"),
                     "profit_per_account": float((realised * approved).mean()), "low": low, "high": high,
                     "versus_approve_all": float((realised * approved).mean() - realised.mean()), "versus_low": gap_low, "versus_high": gap_high,
                     "versus_scorecard": float((realised * approved).mean() - (realised * approvals["scorecard"]).mean()),
                     "versus_scorecard_low": against_low, "versus_scorecard_high": against_high})
    return pd.DataFrame(rows)


def run(settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    out = config.PROCESSED
    out.mkdir(parents=True, exist_ok=True)

    accounts = build.load_accounts()
    if settings.sample:
        accounts = accounts.sample(settings.sample, random_state=config.SEED).reset_index(drop=True)
    y_all = accounts["default"].to_numpy()
    development, holdout = models.holdout_split(accounts)
    dev, hold = accounts.iloc[development].reset_index(drop=True), accounts.iloc[holdout].reset_index(drop=True)
    y_dev, y_hold = y_all[development], y_all[holdout]
    print(f"[analyze] {len(accounts):,} accounts: {len(dev):,} development, {len(hold):,} hold-out", flush=True)

    summary: dict = {"settings": {"cv_folds": settings.cv_folds, "cv_repeats": settings.cv_repeats, "tuning_trials": settings.tuning_trials,
                                  "bootstrap_draws": settings.bootstrap_draws, "sample": settings.sample},
                     "data": {"accounts": len(accounts), "defaults": int(y_all.sum()), "default_rate": float(y_all.mean()),
                              "development": len(dev), "holdout": len(hold), "default_rate_development": float(y_dev.mean()),
                              "default_rate_holdout": float(y_hold.mean())}}
    segments = segment_table(accounts)
    segments.to_csv(out / "segments.csv", index=False, lineterminator="\n")

    # --- the challenger's settings, then cross-validation of every model on the development set
    params, tuning = models.tune_challenger(dev, y_dev, dev["group"].to_numpy(), build.BEHAVIOUR, trials=settings.tuning_trials, folds=settings.cv_folds)
    tuning.to_csv(out / "challenger_tuning.csv", index=False, lineterminator="\n")
    (out / "challenger_params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8", newline="\n")
    make = factories(params)
    print("[analyze] challenger settings chosen; cross-validating", flush=True)
    cv = models.cross_validate(make, dev, y_dev, dev["group"].to_numpy(), settings.cv_folds, settings.cv_repeats)
    cv.to_csv(out / "cv_results.csv", index=False, lineterminator="\n")
    scorecard_variants(dev, y_dev, settings.cv_folds).to_csv(out / "scorecard_variants.csv", index=False, lineterminator="\n")
    cv_summary = cv.groupby("model")[list(validation.METRICS)].agg(["mean", "std"])
    cv_summary.columns = [f"{m}_{s}" for m, s in cv_summary.columns]
    cv_summary.reset_index().to_csv(out / "cv_summary.csv", index=False, lineterminator="\n")

    # --- out-of-fold predictions on the development set (for choosing cut-offs) and the final fits
    oof = {name: models.out_of_fold(make[name], dev, y_dev, dev["group"].to_numpy(), settings.cv_folds) for name in DECISION_MODELS}
    platt = models.Platt().fit(oof["challenger"], y_dev)
    oof["challenger"] = np.round(platt.predict(oof["challenger"]), PROBABILITY_DECIMALS)
    fitted = {name: factory().fit(dev, y_dev) for name, factory in make.items()}
    preds = {name: fitted[name].predict_proba(hold) for name in fitted}
    raw_challenger = preds["challenger"].copy()
    preds["challenger"] = np.round(platt.predict(raw_challenger), PROBABILITY_DECIMALS)
    card: scorecard.Scorecard = fitted["scorecard"]
    summary["challenger"] = {"params": params, "platt_intercept": platt.intercept_, "platt_slope": platt.slope_,
                             "log_loss_before_recalibration": validation.log_loss(y_hold, raw_challenger),
                             "log_loss_after_recalibration": validation.log_loss(y_hold, preds["challenger"])}

    # --- the scorecard itself
    card.screening_.to_csv(out / "scorecard_screening.csv", index=False, lineterminator="\n")
    points = card.points_table()
    points.to_csv(out / "scorecard_points.csv", index=False, lineterminator="\n")
    summary["scorecard"] = {
        "features": list(card.features_), "n_features": len(card.features_), "n_bins": int(len(points)), "intercept": card.intercept_,
        "factor": card.scaling.factor, "offset": card.scaling.offset, "base_score": card.scaling.base_score,
        "base_odds": card.scaling.base_odds, "pdo": card.scaling.pdo, "candidates": int(len(card.screening_)),
        "points_range": {k: int(v) for k, v in card.points_range().items()},
        "iv": {k: float(v) for k, v in card.screening_.set_index("feature")["iv"].items()},
        "rounding_auc_change": validation.auc(y_hold, card.predict_proba(hold)) - validation.auc(y_hold, card.predict_proba_regression(hold)),
        "score_min": int(card.score(hold).min()), "score_max": int(card.score(hold).max())}

    # --- hold-out results
    metrics, comparisons = holdout_table(y_hold, preds, settings.bootstrap_draws)
    metrics.to_csv(out / "holdout_metrics.csv", index=False, lineterminator="\n")
    comparisons.to_csv(out / "holdout_comparisons.csv", index=False, lineterminator="\n")
    cal, lift, roc = [], [], []
    for name in DECISION_MODELS:
        cal.append(validation.calibration_table(y_hold, preds[name]).reset_index().assign(model=name))
        lift.append(validation.lift_table(y_hold, preds[name]).reset_index().assign(model=name))
        roc.append(validation.roc_curve(y_hold, preds[name]).assign(model=name))
    pd.concat(cal).to_csv(out / "calibration.csv", index=False, lineterminator="\n")
    pd.concat(lift).to_csv(out / "lift.csv", index=False, lineterminator="\n")
    pd.concat(roc).to_csv(out / "roc.csv", index=False, lineterminator="\n")
    psi = psi_table(card, dev, hold)
    psi.to_csv(out / "stability_psi.csv", index=False, lineterminator="\n")
    score_hold = card.score(hold)
    edges = np.arange(math.floor(score_hold.min() / 10) * 10, score_hold.max() + 11, 10)
    histogram = pd.DataFrame({"score_from": edges[:-1], "good": np.histogram(score_hold[y_hold == 0], edges)[0],
                              "bad": np.histogram(score_hold[y_hold == 1], edges)[0]})
    histogram.to_csv(out / "score_distribution.csv", index=False, lineterminator="\n")
    balance_hold = economics.exposure(hold)
    risk_bands(y_hold, preds["scorecard"], balance_hold).to_csv(out / "risk_bands.csv", index=False, lineterminator="\n")
    subsets = subset_metrics(hold, y_hold, preds)
    subsets.to_csv(out / "subset_metrics.csv", index=False, lineterminator="\n")
    (out / "examples.json").write_text(json.dumps(_clean(example_accounts(card, hold, y_hold, preds["scorecard"])), indent=2) + "\n", encoding="utf-8", newline="\n")
    summary["psi"] = {"score": float(psi.iloc[0]["psi"]), "max_feature": float(psi.iloc[1:]["psi"].max())}
    summary["cv"] = {row["model"]: {k: float(v) for k, v in row.items() if k != "model"} for row in cv_summary.reset_index().to_dict("records")}

    # --- what the challenger sees
    contributions = fitted["challenger"].contributions(hold)
    importance = pd.DataFrame({"feature": contributions.columns, "mean_abs_shap": contributions.abs().mean().to_numpy(),
                               "direction": [stats.spearmanr(hold[c], contributions[c])[0] for c in contributions.columns]})
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(out / "challenger_importance.csv", index=False, lineterminator="\n")

    # --- the loss on a missed payment, from the roll rates of the development accounts' own repayment history
    evidence_table, roll = rollrates.evidence(dev, y_dev, draws=settings.bootstrap_draws)
    evidence_table.to_csv(out / "rollrates_evidence.csv", index=False, lineterminator="\n")
    rollrates.matrix_table(roll["counts"]).to_csv(out / "rollrates_matrix.csv", index=False, lineterminator="\n")
    rollrates.horizon_table(roll["matrix"], roll["entry_october"]).to_csv(out / "rollrates_horizons.csv", index=False, lineterminator="\n")
    roll["split"].to_csv(out / "rollrates_split.csv", index=False, lineterminator="\n")
    base_evidence = evidence_table[evidence_table["write_off_status"] == config.ECON_WRITEOFF_STATUS].iloc[0]
    summary["rollrates"] = {
        "months": list(rollrates.MONTHS), "transitions": int(roll["counts"].sum()), "p_current_to_late": roll["p_current_to_late"],
        "p_cure_from_first_late": roll["p_cure_from_first_late"], "p_stay_first_late": roll["p_stay_first_late"],
        "p_worse_from_first_late": roll["p_worse_from_first_late"], "entry": roll["entry"].tolist(), "entry_october": roll["entry_october"].tolist(),
        "write_off_status": config.ECON_WRITEOFF_STATUS,
        "horizon": config.ECON_HORIZON, "lgd": config.ECON_LGD, "collection_cost": config.ECON_COLLECTION_COST,
        "base_probability": float(base_evidence["probability"]), "base_loss": float(base_evidence["loss"]), "loss_stress": config.ECON_LOSS_STRESS}

    # --- the economics: the cut-off the data support, and what it earns
    policy = macro.policy_rate()
    base = economics.Economics(revolve_share=economics.revolve_share(dev), funding=economics.policy_rate_on(str(policy.iloc[-1]["effective_date"].date()), policy),
                               loss=float(base_evidence["loss"]))
    balance_dev = economics.exposure(dev)
    tables_hold = {name: economics.threshold_table(y_hold, preds[name], balance_hold) for name in DECISION_MODELS}
    tables_dev = {name: economics.threshold_table(y_dev, oof[name], balance_dev) for name in DECISION_MODELS}
    cutoff_rows = []
    for name in DECISION_MODELS:
        empirical = economics.best_cutoff(economics.profit_curve(tables_dev[name], base))
        for rule, cutoff in (("break-even probability", base.break_even), ("best on the development accounts", float(empirical["cutoff"])),
                             ("approve everyone", 1.0)):
            result = economics.profit_at(tables_hold[name], base, cutoff)
            cutoff_rows.append({"model": name, "rule": rule, "cutoff": cutoff, "approved_share": result["approved_share"],
                                "default_rate_approved": result["default_rate_approved"],
                                "profit_per_account": result["realised_profit"] / len(hold)})
    pd.DataFrame(cutoff_rows).to_csv(out / "cutoffs.csv", index=False, lineterminator="\n")
    curves = []
    for name in DECISION_MODELS:
        curve = economics.profit_curve(tables_hold[name], base)
        curves.append(curve.assign(model=name, profit_per_account=curve["realised_profit"] / len(hold)))
    pd.concat(curves).to_csv(out / "profit_curve.csv", index=False, lineterminator="\n")
    scen = economics.scenarios(tables_hold["scorecard"], base, config.POLICY_SCENARIOS, policy)
    scen.to_csv(out / "scenarios.csv", index=False, lineterminator="\n")
    loss_checks = sorted(float(v) for v in evidence_table["loss"]) + list(config.STRESS_LOSSES)
    sens = economics.sensitivity(tables_hold["scorecard"], base, loss_checks, config.APR_CHECKS)
    sens.to_csv(out / "sensitivity.csv", index=False, lineterminator="\n")

    labels = {float(r.loss): f"Roll rates, written off at status {int(r.write_off_status)}" + (" (base case)" if int(r.write_off_status) == config.ECON_WRITEOFF_STATUS else "")
              for r in evidence_table.itertuples()} | {float(v): f"Stress: {v:.0%} of the balance" for v in config.STRESS_LOSSES}
    case_rows = []
    for loss in loss_checks:
        case = base.with_(loss=loss)
        case_approvals = {"approve_all": np.ones(len(hold), dtype=bool), "scorecard": preds["scorecard"] <= case.break_even, "challenger": preds["challenger"] <= case.break_even}
        case_rows.append(strategy_profits(y_hold, balance_hold, case_approvals, case, settings.bootstrap_draws).assign(case=labels[loss], loss=loss, break_even=case.break_even))
    pd.concat(case_rows).to_csv(out / "loss_cases.csv", index=False, lineterminator="\n")

    cutoff_score = int(math.ceil(float(card.scaling.score(base.break_even)))) if base.break_even > 0 else None
    approvals = {"approve_all": np.ones(len(hold), dtype=bool),
                 "scorecard": preds["scorecard"] <= base.break_even,
                 "challenger": preds["challenger"] <= base.break_even,
                 "last_month_on_time": (hold["status_sep"] <= 0).to_numpy(),
                 "naive_model": preds["naive"] <= base.break_even}
    strategies = strategy_profits(y_hold, balance_hold, approvals, base, settings.bootstrap_draws)
    strategies.to_csv(out / "strategies.csv", index=False, lineterminator="\n")
    summary["economics"] = {
        "apr": base.apr, "revolve_share": base.revolve_share, "yield": base.yield_, "funding": base.funding, "opex": base.opex, "loss": base.loss,
        "margin_month": base.margin, "break_even_pd": base.break_even, "cutoff_score": cutoff_score,
        "cost_ratio": base.loss / base.margin if base.margin > 0 else None,
        "mean_balance": float(balance_hold.mean()), "share_no_balance": float((balance_hold == 0).mean()),
        "policy_rate_date": str(policy.iloc[-1]["effective_date"].date())}

    # --- the explorer behind the dashboard: the threshold table of each model, and the groups inside the scorecard's rows
    explorer = pd.concat([tables_hold[n].assign(model=n).reset_index(names="bin") for n in DECISION_MODELS])
    explorer.to_csv(out / "explorer_bins.csv", index=False, lineterminator="\n")
    bin_of = np.searchsorted(tables_hold["scorecard"]["probability"].to_numpy(), preds["scorecard"], side="left")
    group_rows = []
    for attribute in config.PROTECTED:
        for level, part in hold.assign(_bin=bin_of).groupby(attribute):
            counts = part.groupby("_bin")["default"].agg(["size", "sum"]).reindex(range(len(tables_hold["scorecard"])), fill_value=0)
            group_rows += [{"attribute": attribute, "group": level, "bin": int(b), "accounts": int(r["size"]), "defaults": int(r["sum"])}
                           for b, r in counts.iterrows()]
    pd.DataFrame(group_rows).to_csv(out / "fairness_bins.csv", index=False, lineterminator="\n")

    # --- fairness of the decisions at the break-even cut-off
    frames = {}
    for name in ("scorecard", "challenger"):
        frames[name] = hold[["default", *config.PROTECTED]].assign(probability=preds[name])
    fair = pd.concat([fairness.audit(frames[name], config.PROTECTED, base.break_even, draws=settings.bootstrap_draws).assign(model=name) for name in frames],
                     ignore_index=True)
    fair.to_csv(out / "fairness_groups.csv", index=False, lineterminator="\n")
    proxy = pd.DataFrame([
        {"attribute": "Sex (female)", "auc": fairness.proxy_strength(dev, (dev["sex_code"] == 2).astype(int), build.BEHAVIOUR)},
        {"attribute": "Age 40 or over", "auc": fairness.proxy_strength(dev, (dev["age"] >= 40).astype(int), build.BEHAVIOUR)},
        {"attribute": "University or higher", "auc": fairness.proxy_strength(dev, (dev["education_code"] <= 2).astype(int), build.BEHAVIOUR)}])
    proxy.to_csv(out / "fairness_proxy.csv", index=False, lineterminator="\n")
    judged = fair[(fair["model"] == "scorecard") & (fair["accounts"] >= config.MIN_GROUP_SIZE)]
    summary["fairness"] = {"flagged": [f"{r.attribute}: {r.group}" for r in judged.itertuples() if r.flagged],
                           "lowest_disparate_impact": float(judged["disparate_impact"].min()),
                           "worst_calibration_ratio": float((judged["calibration_ratio"] - 1).abs().max() + 1)}

    # --- what drives the approval gaps: the scorecard adds up points, so each group's gap splits by feature
    points_hold = card.points(hold)
    driver_rows = []
    for attribute in config.PROTECTED:
        part = fair[(fair["model"] == "scorecard") & (fair["attribute"] == attribute) & (fair["accounts"] >= config.MIN_GROUP_SIZE)]
        reference = part.loc[part["approval_rate"].idxmax()]
        gaps = fairness.point_gaps(points_hold, hold[attribute], reference["group"], [g for g in part["group"] if g != reference["group"]])
        gaps = gaps.merge(part[["group", "approval_rate", "disparate_impact", "default_rate", "flagged"]], on="group")
        driver_rows.append(gaps.assign(attribute=attribute, reference_approval_rate=reference["approval_rate"], reference_default_rate=reference["default_rate"]))
    pd.concat(driver_rows).to_csv(out / "fairness_drivers.csv", index=False, lineterminator="\n")

    # --- the second layer: an explainable boosting model, swap sets, stress tests, pricing and re-anchoring to another lender
    def mono_make() -> models.Challenger:
        return models.Challenger(build.BEHAVIOUR, params, monotone=config.MONOTONE)

    oof_mono = models.out_of_fold(mono_make, dev, y_dev, dev["group"].to_numpy(), settings.cv_folds)
    platt_mono = models.Platt().fit(oof_mono, y_dev)
    mono = mono_make().fit(dev, y_dev)
    preds_mono = np.round(platt_mono.predict(mono.predict_proba(hold)), PROBABILITY_DECIMALS)
    adv_preds = {"scorecard": preds["scorecard"], "monotone": preds_mono, "challenger": preds["challenger"]}
    adv_metrics, adv_comparisons = holdout_table(y_hold, adv_preds, settings.bootstrap_draws, comparisons=(("monotone", "scorecard"), ("challenger", "monotone")))
    adv_metrics.to_csv(out / "advanced_models.csv", index=False, lineterminator="\n")
    adv_comparisons.to_csv(out / "advanced_comparisons.csv", index=False, lineterminator="\n")
    adv_approvals = {"approve_all": np.ones(len(hold), dtype=bool), **{name: adv_preds[name] <= base.break_even for name in adv_preds}}
    strategy_profits(y_hold, balance_hold, adv_approvals, base, settings.bootstrap_draws).to_csv(out / "advanced_profits.csv", index=False, lineterminator="\n")
    violations = pd.concat([advanced.monotone_violations(mono, hold, config.MONOTONE).assign(model="monotone"),
                            advanced.monotone_violations(fitted["challenger"], hold, config.MONOTONE).assign(model="challenger")])
    violations.to_csv(out / "monotone_check.csv", index=False, lineterminator="\n")
    summary["monotone"] = {"features_constrained": len(config.MONOTONE), "violations": int(violations.loc[violations["model"] == "monotone", "violations"].sum()),
                           "violations_unconstrained": int(violations.loc[violations["model"] == "challenger", "violations"].sum()),
                           "accounts_checked": int(violations["accounts"].iloc[0])}
    pair_names = {"scorecard": "scorecard", "monotone": "monotone boosting", "challenger": "gradient boosting"}
    swaps = [advanced.swap_sets(y_hold, balance_hold, adv_approvals[a], adv_approvals[b], base, (pair_names[a], pair_names[b])).assign(pair=f"{pair_names[a]} against {pair_names[b]}")
             for a, b in (("scorecard", "challenger"), ("scorecard", "monotone"), ("monotone", "challenger"))]
    pd.concat(swaps).to_csv(out / "swap_sets.csv", index=False, lineterminator="\n")
    advanced.stress_table(preds["scorecard"], balance_hold, base, config.STRESS_SCENARIOS, policy).to_csv(out / "stress.csv", index=False, lineterminator="\n")
    advanced.pricing_table(preds["scorecard"], balance_hold, base).to_csv(out / "pricing.csv", index=False, lineterminator="\n")
    advanced.local_rate_table(preds["scorecard"], balance_hold, base, card.scaling, config.LOCAL_DEFAULT_RATES, float(y_dev.mean())).to_csv(
        out / "local_rate.csv", index=False, lineterminator="\n")

    # --- Pakistan context
    npl, credit = macro.npl_ratio(), macro.private_credit()
    npl.to_csv(out / "pakistan_npl_ratio.csv", index=False, lineterminator="\n")
    credit.to_csv(out / "pakistan_private_credit.csv", index=False, lineterminator="\n")
    summary["pakistan"] = macro.context()

    summary["metrics"] = {row["model"]: {k: float(v) for k, v in row.items() if k != "model"} for row in metrics.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[analyze] scorecard AUC {summary['metrics']['scorecard']['auc']:.3f}, gradient boosting {summary['metrics']['challenger']['auc']:.3f}, "
          f"naive {summary['metrics']['naive']['auc']:.3f}; break-even probability {base.break_even:.1%}", flush=True)
    return summary


def _clean(value):
    """Strict JSON has no NaN or infinity, and numpy numbers must be plain ones."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if not math.isfinite(float(value)) else float(value)
    return value
