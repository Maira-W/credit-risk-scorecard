"""Checks that a folder of results is internally consistent. Used on the committed results and on a small fresh run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditrisk import config, economics
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
    assert len(sens) == len(config.APR_CHECKS) * len(config.LOSS_CHECKS)
    for _, part in sens.groupby("apr"):
        assert part.sort_values("loss")["break_even_pd"].is_monotonic_decreasing
    for _, part in sens.groupby("loss"):
        assert part.sort_values("apr")["break_even_pd"].is_monotonic_increasing

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
            assert (judged["disparate_impact"] == pytest.approx(judged["approval_rate"] / judged["approval_rate"].max()))
            assert (judged["flagged"] == (judged["disparate_impact"] < config.DISPARATE_IMPACT_MIN)).all()
        assert part.loc[part["accounts"] < config.MIN_GROUP_SIZE, "flagged"].eq(False).all()
    proxy = read(folder, "fairness_proxy.csv")
    assert ((proxy["auc"] > 0.4) & (proxy["auc"] < 1)).all()

    # --- descriptive tables
    seg = read(folder, "segments.csv")
    for variable, part in seg.groupby("variable"):
        assert part["share"].sum() == pytest.approx(1.0, abs=1e-9), variable
        assert (part["default_rate"] == pytest.approx(part["defaults"] / part["accounts"]))
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
    return s
