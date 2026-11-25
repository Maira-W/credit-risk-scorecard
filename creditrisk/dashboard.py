"""The interactive dashboard: one HTML file with the data inside, Plotly from a CDN, and no server."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import config, macro

PAGE = config.TEMPLATES / "dashboard.html"
ROC_POINTS = 120
IMPORTANCE_ROWS = 15


def _records(frame: pd.DataFrame, digits: int = 5) -> list[dict]:
    return json.loads(frame.round(digits).to_json(orient="records"))


def _clean(value):
    """Strict JSON has no NaN or infinity, so a number that could not be worked out travels as null."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED / name, **kwargs)


def payload() -> dict:
    p = config.PROCESSED
    summary = json.loads((p / "summary.json").read_text(encoding="utf-8"))
    points = _read("scorecard_points.csv")
    points["name"] = points["feature"].map(config.feature_name)
    screening = _read("scorecard_screening.csv")
    screening["name"] = screening["feature"].map(config.feature_name)
    roc = _read("roc.csv")
    roc_out = {}
    for model, part in roc.groupby("model"):
        keep = np.unique(np.linspace(0, len(part) - 1, ROC_POINTS).astype(int))
        roc_out[model] = {"fpr": [round(float(v), 4) for v in part["fpr"].iloc[keep]], "tpr": [round(float(v), 4) for v in part["tpr"].iloc[keep]]}
    explorer = _read("explorer_bins.csv")
    explorer_out = {m: _records(part[["probability", "accounts", "defaults", "balance_paid", "balance_defaulted", "balance"]], 9)
                    for m, part in explorer.groupby("model", sort=False)}
    fair_bins = _read("fairness_bins.csv")
    fair_out = [{"attribute": a, "group": g, "accounts": part["accounts"].tolist(), "defaults": part["defaults"].tolist()}
                for (a, g), part in fair_bins.groupby(["attribute", "group"], sort=False)]
    fairness = _read("fairness_groups.csv")
    importance = _read("challenger_importance.csv").head(IMPORTANCE_ROWS)
    importance["name"] = importance["feature"].map(config.feature_name)
    sensitivity = _read("sensitivity.csv")
    policy = macro.policy_rate()
    policy["effective_date"] = policy["effective_date"].dt.strftime("%Y-%m-%d")
    return _clean({
        "stats": summary,
        "segments": _records(_read("segments.csv"), 5),
        "points": _records(points, 4),
        "screening": _records(screening, 4),
        "metrics": _records(_read("holdout_metrics.csv"), 5),
        "comparisons": _records(_read("holdout_comparisons.csv"), 5),
        "cv": _records(_read("cv_summary.csv"), 5),
        "roc": roc_out,
        "calibration": _records(_read("calibration.csv"), 5),
        "lift": _records(_read("lift.csv"), 5),
        "scoreDist": _records(_read("score_distribution.csv"), 0),
        "bands": _records(_read("risk_bands.csv"), 5),
        "importance": _records(importance, 5),
        "psi": _records(_read("stability_psi.csv"), 5),
        "explorer": explorer_out,
        "strategies": _records(_read("strategies.csv"), 5),
        "scenarios": _records(_read("scenarios.csv"), 5),
        "fairGroups": _records(fairness[fairness["model"] == "scorecard"], 5),
        "fairBins": fair_out,
        "proxy": _records(_read("fairness_proxy.csv"), 4),
        "npl": _records(_read("pakistan_npl_ratio.csv"), 3),
        "credit": _records(_read("pakistan_private_credit.csv"), 3),
        "policy": _records(policy[["effective_date", "policy_rate_pct"]], 2),
        "scenarioDates": list(config.POLICY_SCENARIOS),
        "aprChecks": list(config.APR_CHECKS), "lossChecks": sorted(float(v) for v in sensitivity["loss"].unique()),
        "rollMatrix": _records(_read("rollrates_matrix.csv"), 5), "rollEvidence": _records(_read("rollrates_evidence.csv"), 5), "rollSplit": _records(_read("rollrates_split.csv"), 5),
        "lossCases": _records(_read("loss_cases.csv"), 5), "advModels": _records(_read("advanced_models.csv"), 5),
        "advComparisons": _records(_read("advanced_comparisons.csv"), 5), "advProfits": _records(_read("advanced_profits.csv"), 5),
        "monotone": _records(_read("monotone_check.csv"), 5), "watchBelow": config.DISPARATE_IMPACT_WATCH, "swap": _records(_read("swap_sets.csv"), 5), "stress": _records(_read("stress.csv"), 5),
        "pricing": _records(_read("pricing.csv"), 5), "localRate": _records(_read("local_rate.csv"), 5), "drivers": _records(_read("fairness_drivers.csv"), 4),
        "featureNames": {c: config.feature_name(c) for c in sorted(set(points["feature"]) | set(importance["feature"]))},
        "statusMeaning": {str(k): v for k, v in config.STATUS_MEANING.items()},
        "source": {"mcb": config.MCB_CARD_URL, "uci": config.UCI_PAGE},
    })


def render() -> str:
    page = PAGE.read_text(encoding="utf-8")
    data = json.dumps(payload(), separators=(",", ":"), allow_nan=False)
    return page.replace("__PLOTLY__", config.PLOTLY_JS).replace("__DATA__", data.replace("</", "<\\/"))


def build() -> None:
    config.DASHBOARD.mkdir(parents=True, exist_ok=True)
    (config.DASHBOARD / "dashboard.html").write_text(render(), encoding="utf-8", newline="\n")
    (config.ROOT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=dashboard/dashboard.html">'
        '<title>Credit card default scorecard</title><p><a href="dashboard/dashboard.html">Open the dashboard</a></p></html>\n',
        encoding="utf-8", newline="\n")
    print(f"[dashboard] {(config.DASHBOARD / 'dashboard.html').stat().st_size / 1e3:,.0f} kB")
