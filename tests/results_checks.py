"""Checks that a folder of results is internally consistent. Used on the committed results and on a small fresh run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditrisk import config, economics, rollrates
from creditrisk.scorecard import Scaling

MODELS = {"naive", "scorecard", "scorecard_demographics", "challenger", "challenger_demographics"}
DECISION = ("naive", "scorecard", "challenger")
SLACK = 0.01


def read(folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(folder / name)


def check_results(folder: Path) -> dict:
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    n_dev, n_hold = s["data"]["development"], s["data"]["holdout"]
    assert s["data"]["accounts"] == n_dev + n_hold

    # --- hold-out metrics
    m = read(folder, "holdout_metrics.csv").set_index("model")
    assert set(m.index) == MODELS
    for metric in ("auc", "ks", "brier"):
        assert ((m[f"{metric}_low"] - SLACK <= m[metric]) & (m[metric] <= m[f"{metric}_high"] + SLACK)).all(), metric
        assert (m[f"{metric}_low"] < m[f"{metric}_high"]).all()
    assert ((m["auc"] > 0.5) & (m["auc"] < 1)).all() and ((m["ks"] > 0) & (m["ks"] < 1)).all()
    assert (m["gini"] - (2 * m["auc"] - 1)).abs().max() < 1e-9
    assert m.loc["scorecard", "auc"] > m.loc["naive", "auc"]
    assert s["metrics"]["scorecard"]["auc"] == pytest.approx(m.loc["scorecard", "auc"])
    comparisons = read(folder, "holdout_comparisons.csv")
    assert len(comparisons) == 4 * 3
    for r in comparisons.itertuples():
        assert r.low - SLACK <= r.difference <= r.high + SLACK
        assert r.difference == pytest.approx(m.loc[r.model, r.metric] - m.loc[r.against, r.metric])

    # --- calibration, lift, ROC
    for name, total in (("calibration.csv", "accounts"), ("lift.csv", "accounts")):
        table = read(folder, name)
        for model in DECISION:
            part = table[table["model"] == model]
            assert len(part) == config.DECILES and part[total].sum() == n_hold
    lift = read(folder, "lift.csv")
    for model in DECISION:
        part = lift[lift["model"] == model]
        assert part["capture"].iloc[-1] == pytest.approx(1.0) and part["defaults"].sum() == round(s["data"]["default_rate_holdout"] * n_hold)
        assert part["capture"].is_monotonic_increasing
    roc = read(folder, "roc.csv")
    for model in DECISION:
        part = roc[roc["model"] == model]
        assert (part.iloc[0][["fpr", "tpr"]] == 0).all() and (part.iloc[-1][["fpr", "tpr"]] == 1).all()
        assert part["fpr"].is_monotonic_increasing and part["tpr"].is_monotonic_increasing

    # --- the scorecard
    points = read(folder, "scorecard_points.csv")
    assert list(dict.fromkeys(points["feature"])) == s["scorecard"]["features"]
    assert points["points"].dtype.kind == "i" and len(points) == s["scorecard"]["n_bins"]
    assert (points.groupby("feature")["accounts"].sum() == n_dev).all()
    screening = read(folder, "scorecard_screening.csv")
    assert (screening["decision"] == "kept").sum() == s["scorecard"]["n_features"] and len(screening) == s["scorecard"]["candidates"]
    assert (screening.loc[screening["decision"] == "dropped", "reason"].fillna("") != "").all()
    assert (screening.loc[screening["decision"] == "kept", "coefficient"] > 0).all()
    low = points.groupby("feature")["points"].min().sum()
    high = points.groupby("feature")["points"].max().sum()
    assert low <= s["scorecard"]["score_min"] and s["scorecard"]["score_max"] <= high
    dist = read(folder, "score_distribution.csv")
    assert dist["good"].sum() + dist["bad"].sum() == n_hold

    # --- the economics
    e = s["economics"]
    margin = (e["apr"] * e["revolve_share"] - e["funding"] - e["opex"]) / 12
    assert e["margin_month"] == pytest.approx(margin) and e["break_even_pd"] == pytest.approx(margin / (margin + e["loss"]))
    assert 0 < e["revolve_share"] < 1
    cuts = read(folder, "cutoffs.csv")
    assert len(cuts) == 9
    everyone = cuts[cuts["rule"] == "approve everyone"]
    assert (everyone["approved_share"] == 1).all() and everyone["profit_per_account"].nunique() == 1
    strat = read(folder, "strategies.csv").set_index("rule")
    assert strat.loc["approve_all", "versus_approve_all"] == 0 and strat.loc["approve_all", "profit_per_account"] == pytest.approx(everyone["profit_per_account"].iloc[0])
    assert ((strat["low"] - 1 <= strat["profit_per_account"]) & (strat["profit_per_account"] <= strat["high"] + 1)).all()
    assert strat.loc["scorecard", "versus_scorecard"] == 0 and strat.loc["scorecard", "versus_scorecard_low"] == 0
    for rule in strat.index:
        assert strat.loc[rule, "versus_scorecard"] == pytest.approx(strat.loc[rule, "profit_per_account"] - strat.loc["scorecard", "profit_per_account"])
        assert strat.loc[rule, "versus_approve_all"] == pytest.approx(strat.loc[rule, "profit_per_account"] - strat.loc["approve_all", "profit_per_account"])
        assert strat.loc[rule, "versus_low"] - 1 <= strat.loc[rule, "versus_approve_all"] <= strat.loc[rule, "versus_high"] + 1
    scen = read(folder, "scenarios.csv")
    assert len(scen) == len(config.POLICY_SCENARIOS)
    assert scen["profit_per_account"].iloc[0] == pytest.approx(scen["profit_per_account_first_cutoff_kept"].iloc[0])
    by_rate = scen.sort_values("policy_rate")
    assert by_rate["break_even_pd"].is_monotonic_decreasing and by_rate["approved_share"].is_monotonic_decreasing
    policy = pd.read_csv(config.POLICY_RATE_FILE, parse_dates=["effective_date"])
    for r in scen.itertuples():
        assert r.policy_rate == pytest.approx(economics.policy_rate_on(r.date, policy))
    sens = read(folder, "sensitivity.csv")
    assert len(sens) == len(config.APR_CHECKS) * (len(config.WRITEOFF_CHECKS) + len(config.STRESS_LOSSES))
    for _, part in sens.groupby("apr"):
        assert part.sort_values("loss")["break_even_pd"].is_monotonic_decreasing
    for _, part in sens.groupby("loss"):
        assert part.sort_values("apr")["break_even_pd"].is_monotonic_increasing
    at_base = sens[np.isclose(sens["apr"], e["apr"]) & np.isclose(sens["loss"], e["loss"])]
    scorecard_cut = cuts[(cuts["model"] == "scorecard") & (cuts["rule"] == "break-even probability")].iloc[0]
    assert len(at_base) == 1 and at_base["profit_per_account"].iloc[0] == pytest.approx(scorecard_cut["profit_per_account"])      # the grid contains the base case

    # --- the explorer behind the dashboard reproduces the cut-off results exactly
    explorer = read(folder, "explorer_bins.csv")
    econ = economics.Economics(apr=e["apr"], revolve_share=e["revolve_share"], funding=e["funding"], opex=e["opex"], loss=e["loss"])
    for model in DECISION:
        part = explorer[explorer["model"] == model].drop(columns=["model", "bin"]).reset_index(drop=True)
        assert part["accounts"].sum() == n_hold and part["probability"].is_monotonic_increasing and part["probability"].is_unique
        result = economics.profit_at(part, econ, econ.break_even)
        row = cuts[(cuts["model"] == model) & (cuts["rule"] == "break-even probability")].iloc[0]
        assert result["approved_share"] == pytest.approx(row["approved_share"]) and result["realised_profit"] / n_hold == pytest.approx(row["profit_per_account"])
    score_probability = explorer[explorer["model"] == "scorecard"]["probability"].to_numpy()
    scaling = Scaling(s["scorecard"]["base_score"], s["scorecard"]["base_odds"], s["scorecard"]["pdo"])
    scores = np.rint(scaling.score(score_probability))
    assert np.allclose(scaling.probability(scores), score_probability, atol=1e-12)          # every scorecard probability belongs to a whole-number score

    # --- fairness
    groups = read(folder, "fairness_groups.csv")
    for (model, attribute), part in groups.groupby(["model", "attribute"]):
        assert part["accounts"].sum() == n_hold and part["share"].sum() == pytest.approx(1.0), (model, attribute)
    card = groups[groups["model"] == "scorecard"]
    bins = read(folder, "fairness_bins.csv")
    table = explorer[explorer["model"] == "scorecard"].reset_index(drop=True)
    last = int(np.max(np.where(table["probability"].to_numpy() <= e["break_even_pd"])[0])) if (table["probability"] <= e["break_even_pd"]).any() else -1
    for (attribute, group), part in bins.groupby(["attribute", "group"]):
        mine = card[(card["attribute"] == attribute) & (card["group"].astype(str) == str(group))].iloc[0]
        part = part.sort_values("bin")
        assert part["accounts"].sum() == mine["accounts"]
        approved = part["accounts"].iloc[: last + 1].sum()
        assert approved / part["accounts"].sum() == pytest.approx(mine["approval_rate"])
    for _attribute, part in card.groupby("attribute"):
        judged = part[part["accounts"] >= config.MIN_GROUP_SIZE]
        assert part.loc[part["accounts"] < config.MIN_GROUP_SIZE, ["di_low", "di_high"]].isna().all().all()
        assert ((judged["di_low"] - SLACK <= judged["disparate_impact"]) & (judged["disparate_impact"] <= judged["di_high"] + SLACK)).all()
        assert ((judged["di_low"] <= judged["di_high"]) & (judged["di_high"] <= 1.0 + 1e-9)).all()
        if len(judged) and judged["approval_rate"].max() > 0:
            assert np.allclose(judged["disparate_impact"], judged["approval_rate"] / judged["approval_rate"].max())
            assert (judged["flagged"] == (judged["disparate_impact"] < config.DISPARATE_IMPACT_MIN)).all()
        assert part.loc[part["accounts"] < config.MIN_GROUP_SIZE, "flagged"].eq(False).all()
    proxy = read(folder, "fairness_proxy.csv")
    assert ((proxy["auc"] > 0.4) & (proxy["auc"] < 1)).all()

    # --- descriptive tables
    seg = read(folder, "segments.csv")
    for variable, part in seg.groupby("variable"):
        assert part["share"].sum() == pytest.approx(1.0, abs=1e-9), variable
        assert np.allclose(part["default_rate"], part["defaults"] / part["accounts"])
        assert ((part["ci_low"] <= part["default_rate"]) & (part["default_rate"] <= part["ci_high"])).all()
    subsets = read(folder, "subset_metrics.csv")
    assert len(subsets) == 6 and subsets.groupby("subset")["accounts"].first().sum() == n_hold
    psi = read(folder, "stability_psi.csv")
    assert (psi["psi"] >= 0).all() and psi.iloc[0]["item"] == "Score" and psi.iloc[0]["psi"] < 0.1
    bands = read(folder, "risk_bands.csv")
    assert bands["accounts"].sum() == n_hold and bands["share_of_defaults"].sum() == pytest.approx(1.0)
    importance = read(folder, "challenger_importance.csv")
    assert importance["share"].sum() == pytest.approx(1.0) and importance["mean_abs_shap"].is_monotonic_decreasing
    tuning = read(folder, "challenger_tuning.csv")
    chosen = json.loads((folder / "challenger_params.json").read_text(encoding="utf-8"))
    best = tuning.sort_values(["cv_log_loss", "trial"]).iloc[0]
    assert all(best[k] == pytest.approx(v) for k, v in chosen.items())
    cv = read(folder, "cv_results.csv")
    assert set(cv["model"]) == MODELS and cv.groupby("model").size().nunique() == 1
    variants = read(folder, "scorecard_variants.csv")
    assert len(variants) == len(config.CORRELATION_CHECKS) + 1 and variants["variant"].str.contains(r"\(used\)").sum() == 1
    assert variants["cv_auc"].between(0.6, 0.9).all() and variants["features"].between(1, 20).all()
    examples = json.loads((folder / "examples.json").read_text(encoding="utf-8"))
    assert [x["label"] for x in examples][1] == "In the middle" and examples[0]["score"] < examples[1]["score"] < examples[2]["score"]
    check_second_layer(folder, s)
    return s


def check_second_layer(folder: Path, s: dict) -> None:
    """The loss from roll rates, the explainable model, swap sets, stress tests, pricing, re-anchoring and the drivers of the fairness gaps."""
    n_dev, n_hold = s["data"]["development"], s["data"]["holdout"]
    e, roll = s["economics"], s["rollrates"]
    strat = read(folder, "strategies.csv").set_index("rule")

    # --- roll rates: the loss on a missed payment is the formula applied to the measured chance, and the economics use it
    evidence = read(folder, "rollrates_evidence.csv")
    assert list(evidence["write_off_status"]) == list(config.WRITEOFF_CHECKS) and (evidence["months"] == config.ECON_HORIZON).all()
    assert evidence["probability"].between(0, 1).all() and evidence["probability"].is_monotonic_decreasing
    assert ((evidence["low"] - SLACK <= evidence["probability"]) & (evidence["probability"] <= evidence["high"] + SLACK)).all()
    assert np.allclose(evidence["loss"], roll["lgd"] * evidence["probability"] + roll["collection_cost"])
    assert (evidence["loss_low"] <= evidence["loss"] + SLACK).all() and (evidence["loss"] <= evidence["loss_high"] + SLACK).all()
    base_row = evidence[evidence["write_off_status"] == roll["write_off_status"]].iloc[0]
    assert roll["base_loss"] == pytest.approx(base_row["loss"]) and roll["base_probability"] == pytest.approx(base_row["probability"])
    assert e["loss"] == pytest.approx(roll["base_loss"])
    matrix = read(folder, "rollrates_matrix.csv")
    assert len(matrix) == 36 and matrix["moves"].sum() == roll["transitions"] == n_dev * (len(roll["months"]) - 1)
    assert np.allclose(matrix.groupby("from_state")["probability"].sum(), 1.0)
    assert sum(roll["entry"]) == pytest.approx(1.0) and sum(roll["entry_october"]) == pytest.approx(1.0) and roll["entry_october"][0] == 0      # a missed payment is late
    assert roll["p_cure_from_first_late"] + roll["p_stay_first_late"] + roll["p_worse_from_first_late"] == pytest.approx(1.0)
    split = read(folder, "rollrates_split.csv")
    assert len(split) == 2 * len(config.WRITEOFF_CHECKS) and list(dict.fromkeys(split["group"])) == list(rollrates.GROUPS[1:])
    shares = split.drop_duplicates("group").set_index("group")["share"]
    assert shares.sum() == pytest.approx(1.0) and split.groupby("group")["share"].nunique().eq(1).all() and roll["entry"][0] == pytest.approx(shares["Current in September"])
    assert np.allclose(split["loss"], roll["lgd"] * split["probability"] + roll["collection_cost"])
    assert ((split["low"] - SLACK <= split["probability"]) & (split["probability"] <= split["high"] + SLACK)).all()
    current, late = (split[split["group"] == g].set_index("write_off_status") for g in rollrates.GROUPS[1:])
    for row in evidence.itertuples():            # the chance is linear in where the accounts start: the whole is the share-weighted average of its two parts
        whole = shares["Current in September"] * current.loc[row.write_off_status, "probability"] + shares["Late in September"] * late.loc[row.write_off_status, "probability"]
        assert row.probability == pytest.approx(whole)
    horizons = read(folder, "rollrates_horizons.csv")
    assert len(horizons) == 12
    for _, part in horizons.groupby("write_off_status"):
        assert part.sort_values("months")["probability"].is_monotonic_increasing
    for _, part in horizons.groupby("months"):
        assert part.sort_values("write_off_status")["probability"].is_monotonic_decreasing
    twelve = horizons[horizons["months"] == config.ECON_HORIZON].set_index("write_off_status")["probability"]
    assert np.allclose(twelve.loc[evidence["write_off_status"]].to_numpy(), evidence["probability"].to_numpy())

    # --- the loss cases: the base case is the main result, and every case has its own break-even probability
    cases = read(folder, "loss_cases.csv")
    assert set(cases["rule"]) == {"approve_all", "scorecard", "challenger"} and cases["case"].nunique() == len(config.WRITEOFF_CHECKS) + len(config.STRESS_LOSSES)
    for loss, part in cases.groupby("loss"):
        assert np.allclose(part["break_even"], e["margin_month"] / (e["margin_month"] + loss)) and part["case"].nunique() == 1
    base_case = cases[np.isclose(cases["loss"], e["loss"])].set_index("rule")
    assert len(base_case) == 3 and base_case["case"].str.contains("base case").all()
    for rule in base_case.index:
        assert base_case.loc[rule, "profit_per_account"] == pytest.approx(strat.loc[rule, "profit_per_account"])
    assert np.isclose(cases["loss"], config.ECON_LOSS_STRESS).any()                           # the earlier 15% guess is still shown, as a stress

    # --- the explainable model: measured like the others, and it keeps the directions it was given
    adv = read(folder, "advanced_models.csv").set_index("model")
    assert set(adv.index) == {"scorecard", "monotone", "challenger"}
    for metric in ("auc", "ks", "brier"):
        assert ((adv[f"{metric}_low"] - SLACK <= adv[metric]) & (adv[metric] <= adv[f"{metric}_high"] + SLACK)).all(), metric
    main = read(folder, "holdout_metrics.csv").set_index("model")
    for model in ("scorecard", "challenger"):
        assert adv.loc[model, "auc"] == pytest.approx(main.loc[model, "auc"]) and adv.loc[model, "brier"] == pytest.approx(main.loc[model, "brier"])
    assert adv.loc["monotone", "auc"] > main.loc["naive", "auc"]
    pairs = read(folder, "advanced_comparisons.csv")
    assert len(pairs) == 2 * 3
    for row in pairs.itertuples():
        assert row.low - SLACK <= row.difference <= row.high + SLACK
        assert row.difference == pytest.approx(adv.loc[row.model, row.metric] - adv.loc[row.against, row.metric])
    profits = read(folder, "advanced_profits.csv").set_index("rule")
    assert set(profits.index) == {"approve_all", "scorecard", "monotone", "challenger"}
    for rule in ("approve_all", "scorecard", "challenger"):
        assert profits.loc[rule, "profit_per_account"] == pytest.approx(strat.loc[rule, "profit_per_account"])
    assert profits.loc["scorecard", "versus_scorecard"] == 0 and ((profits["low"] - 1 <= profits["profit_per_account"]) & (profits["profit_per_account"] <= profits["high"] + 1)).all()
    check = read(folder, "monotone_check.csv")
    mono, free = check[check["model"] == "monotone"], check[check["model"] == "challenger"]
    assert len(mono) > 0 and list(mono["feature"]) == list(free["feature"]) and set(check["feature"]) <= set(config.MONOTONE)
    assert (check["direction"] == check["feature"].map(config.MONOTONE)).all() and (check["violations"] <= check["accounts"]).all()
    assert (mono["violations"] == 0).all() and (mono["largest_fall"] == 0).all()              # a boosting model told a direction obeys it
    assert check["largest_fall"].between(0, 1).all() and (check["average_fall"] <= check["largest_fall"] + 1e-12).all()
    assert ((check["violations"] > 0) <= (check["largest_fall"] > 0)).all() and ((check["violations"] == 0) <= (check["average_fall"] == 0)).all()
    m = s["monotone"]
    assert (m["violations"], m["violations_unconstrained"], m["accounts_checked"]) == (mono["violations"].sum(), free["violations"].sum(), check["accounts"].iloc[0])

    # --- swap sets: the four cells cover every account, and add up to approving everyone
    swaps = read(folder, "swap_sets.csv")
    assert swaps["pair"].nunique() == 3 and len(swaps) == 12
    for pair, part in swaps.groupby("pair"):
        assert part["accounts"].sum() == n_hold and part["share"].sum() == pytest.approx(1.0), pair
        assert part["profit_if_approved"].sum() == pytest.approx(strat.loc["approve_all", "profit_per_account"]), pair
        assert (part["accounts"] * part["default_rate"].fillna(0)).sum() == pytest.approx(round(s["data"]["default_rate_holdout"] * n_hold)), pair
    cells = swaps[swaps["pair"] == "scorecard against gradient boosting"].set_index("cell")["accounts"]
    assert cells["Both approve"] + cells["Only scorecard approves"] == round(strat.loc["scorecard", "approved_share"] * n_hold)
    assert cells["Both approve"] + cells["Only gradient boosting approves"] == round(strat.loc["challenger", "approved_share"] * n_hold)

    # --- stress tests
    stress = read(folder, "stress.csv")
    assert list(stress["scenario"]) == [x[0] for x in config.STRESS_SCENARIOS] and list(stress["odds_factor"]) == [x[1] for x in config.STRESS_SCENARIOS]
    first = stress.iloc[0]
    assert first["odds_factor"] == 1 and first["policy_rate"] == pytest.approx(e["funding"]) and first["break_even_pd"] == pytest.approx(e["break_even_pd"])
    assert first["profit_kept"] == pytest.approx(first["profit_reset"]) and first["approved_kept"] == pytest.approx(first["approved_reset"], abs=1 / n_hold)
    assert (stress["profit_reset"] >= stress["profit_kept"] - 1e-9).all() and stress["mean_probability"].between(0, 1).all()
    same_rate = stress[np.isclose(stress["policy_rate"], e["funding"])].sort_values("odds_factor")
    assert same_rate["profit_kept"].is_monotonic_decreasing and same_rate["approved_reset"].is_monotonic_decreasing and same_rate["mean_probability"].is_monotonic_increasing

    # --- risk-based pricing: each group's APR is the one that makes it worth exactly nothing
    pricing = read(folder, "pricing.csv")
    p = pricing["mean_probability"]
    assert len(pricing) == config.PRICING_GROUPS and pricing["accounts"].sum() == n_hold
    assert p.is_monotonic_increasing and pricing["required_apr"].is_monotonic_increasing
    assert np.allclose(pricing["required_margin_month"], p * e["loss"] / (1 - p))
    assert np.allclose(pricing["required_apr"], (12 * pricing["required_margin_month"] + e["funding"] + e["opex"]) / e["revolve_share"])
    assert (pricing["pays_at_current_apr"] == (pricing["required_apr"] <= e["apr"])).all()

    # --- re-anchoring to another lender's default rate: at the scorecard's own rate nothing moves
    local = read(folder, "local_rate.csv")
    assert len(local) == len(config.LOCAL_DEFAULT_RATES) + 1
    own = local.sort_values("local_default_rate").iloc[-1]
    assert own["local_default_rate"] == pytest.approx(s["data"]["default_rate_development"]) and own["log_odds_shift"] == pytest.approx(0, abs=1e-12)
    assert own["cutoff_probability_in_original_terms"] == pytest.approx(e["break_even_pd"])
    assert own["approved_share"] == pytest.approx(strat.loc["scorecard", "approved_share"], abs=2 / n_hold)
    ordered = local.sort_values("local_default_rate")
    assert ordered["approved_share"].is_monotonic_decreasing and ordered["cutoff_score"].is_monotonic_increasing

    # --- what drives the fairness gaps: the gaps by feature add up to the gap in the average score
    drivers = read(folder, "fairness_drivers.csv")
    assert set(drivers["attribute"]) <= set(config.PROTECTED) and (drivers["group"].astype(str) != drivers["reference"].astype(str)).all()
    assert (drivers["approval_rate"] <= drivers["reference_approval_rate"] + 1e-12).all()
    for (attribute, group), part in drivers.groupby(["attribute", "group"]):
        assert set(part["feature"]) == set(s["scorecard"]["features"]), (attribute, group)
        assert part["points_gap"].sum() == pytest.approx(part["total_points_gap"].iloc[0]), (attribute, group)
        assert part["disparate_impact"].iloc[0] == pytest.approx(part["approval_rate"].iloc[0] / part["reference_approval_rate"].iloc[0]), (attribute, group)
