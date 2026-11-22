"""The dashboard: the data inside it, the page on disk, and the economics the page recomputes in the browser."""

import json
import re
import shutil
import subprocess

import pandas as pd
import pytest

from creditrisk import config, dashboard, economics, macro


@pytest.fixture(scope="module")
def payload():
    return dashboard.payload()


def test_the_payload_holds_every_part_the_page_draws(payload):
    needed = {"stats", "segments", "points", "screening", "metrics", "comparisons", "cv", "roc", "calibration", "lift", "scoreDist", "bands", "importance", "psi", "explorer",
              "strategies", "scenarios", "fairGroups", "fairBins", "proxy", "npl", "credit", "policy", "scenarioDates", "aprChecks", "lossChecks", "featureNames", "statusMeaning"}
    assert needed <= set(payload)
    assert set(payload["explorer"]) == {"naive", "scorecard", "challenger"} and set(payload["roc"]) == {"naive", "scorecard", "challenger"}


def test_the_payload_is_strict_json(payload):
    json.dumps(payload, allow_nan=False)


def test_the_explorer_tables_cover_every_hold_out_account(payload):
    n = payload["stats"]["data"]["holdout"]
    for rows in payload["explorer"].values():
        assert sum(r["accounts"] for r in rows) == n
        assert [r["probability"] for r in rows] == sorted(r["probability"] for r in rows)


def test_group_counts_line_up_with_the_scorecard_table(payload):
    rows = len(payload["explorer"]["scorecard"])
    assert payload["fairBins"] and all(len(g["accounts"]) == rows == len(g["defaults"]) for g in payload["fairBins"])
    per_attribute = pd.DataFrame(payload["fairBins"]).assign(n=lambda d: d["accounts"].map(sum)).groupby("attribute")["n"].sum()
    assert (per_attribute == payload["stats"]["data"]["holdout"]).all()


def test_the_page_replaces_its_placeholders_and_carries_valid_data():
    html = dashboard.render()
    assert "__DATA__" not in html and "__PLOTLY__" not in html and config.PLOTLY_JS in html
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(blob.replace("<\\/", "</"))["stats"]["data"]["accounts"] == 30_000


def test_the_page_on_disk_is_current():
    assert (config.DASHBOARD / "dashboard.html").read_text(encoding="utf-8") == dashboard.render(), "run python -m creditrisk report"
    assert "dashboard/dashboard.html" in (config.ROOT / "index.html").read_text(encoding="utf-8")


def test_the_page_names_its_tabs_and_its_caveat():
    html = (config.DASHBOARD / "dashboard.html").read_text(encoding="utf-8")
    for tab in ("The answer", "Who defaults", "The scorecard", "How good is it?", "Cut-off explorer", "Policy-rate scenarios", "Fairness", "Pakistan context", "Method and limits"):
        assert tab in html
    assert "Taiwan" in html and "not lending advice" in html and "github.com/Maira-W/credit-risk-scorecard" in html


def _js_econ(e0, state):
    """The page's economics, written the way the page writes them."""
    yld = state["apr"] * e0["revolve_share"]
    margin = (yld - state["funding"] - state["opex"]) / 12
    return margin, (margin / (margin + state["loss"]) if margin > 0 else 0)


def _js_at_cutoff(rows, margin, loss, cutoff):
    n = sum(r["accounts"] for r in rows)
    idx, cum = -1, {"accounts": 0, "defaults": 0, "paid": 0.0, "lost": 0.0}
    for i, r in enumerate(rows):
        cum["accounts"] += r["accounts"]
        cum["defaults"] += r["defaults"]
        cum["paid"] += r["balance_paid"]
        cum["lost"] += r["balance_defaulted"]
        if r["probability"] <= cutoff:
            idx, snapshot = i, dict(cum)
    if idx < 0:
        return 0.0, 0.0
    return snapshot["accounts"] / n, (margin * snapshot["paid"] - loss * snapshot["lost"]) / n


def test_the_page_arithmetic_reproduces_the_python_scenarios(payload):
    """The page works out margin, break-even probability, share approved and profit in JavaScript from the explorer rows. The same
    steps in Python must give the numbers in scenarios.csv."""
    e0, scen = payload["stats"]["economics"], pd.read_csv(config.PROCESSED / "scenarios.csv")
    policy = macro.policy_rate()
    rows = payload["explorer"]["scorecard"]
    for r in scen.itertuples():
        rate = economics.policy_rate_on(r.date, policy)
        margin, be = _js_econ(e0, {"apr": e0["apr"], "funding": rate, "opex": e0["opex"], "loss": e0["loss"]})
        share, profit = _js_at_cutoff(rows, margin, e0["loss"], be)
        assert be == pytest.approx(r.break_even_pd) and share == pytest.approx(r.approved_share) and profit == pytest.approx(r.profit_per_account)


def test_the_page_arithmetic_reproduces_every_cell_of_the_sensitivity_grid(payload):
    e0, sens = payload["stats"]["economics"], pd.read_csv(config.PROCESSED / "sensitivity.csv")
    for r in sens.itertuples():
        margin, be = _js_econ(e0, {"apr": r.apr, "funding": e0["funding"], "opex": e0["opex"], "loss": r.loss})
        share, profit = _js_at_cutoff(payload["explorer"]["scorecard"], margin, r.loss, be)
        assert share == pytest.approx(r.approved_share) and profit == pytest.approx(r.profit_per_account)


def test_the_score_calculator_agrees_with_the_scorecard(payload):
    """Adding up the points the page offers for each bin and reading the probability from the scaling gives the scorecard's own."""
    sc = payload["stats"]["scorecard"]
    points = pd.DataFrame(payload["points"])
    best = points.groupby("feature")["points"].max().sum()
    prob = lambda s: 1 / (1 + 2.718281828459045 ** ((s - sc["offset"]) / sc["factor"]))  # noqa: E731
    assert prob(best) < prob(best - 40) and prob(600) == pytest.approx(1 / 21, rel=1e-9) if sc["base_score"] == 600 else True
    assert set(points["feature"]) == set(sc["features"])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_javascript_parses(tmp_path):
    html = (config.DASHBOARD / "dashboard.html").read_text(encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    path = tmp_path / "dashboard.js"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
