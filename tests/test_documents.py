"""The README and the documents: every placeholder filled, up to date, links that work, and sentences that follow the numbers."""

import re
from pathlib import Path

import pandas as pd
import pytest

from creditrisk import config, docs


@pytest.fixture(scope="module")
def values():
    return docs.values()


def test_render_fills_placeholders_and_formats():
    assert docs.render("{{a}} and {{ b :.1f }} and {{c:,}}", {"a": "x", "b": 2.345, "c": 12345}) == "x and 2.3 and 12,345"


def test_a_placeholder_with_no_value_is_an_error():
    with pytest.raises(KeyError, match="nope"):
        docs.render("{{nope}}", {})


def test_helpers():
    assert docs._join([]) == "none" and docs._join(["a"]) == "a" and docs._join(["a", "b", "c"]) == "a, b and c"
    assert docs._p(0.2212) == "22.1%" and docs._p(0.2212, 0) == "22%" and docs._nt(-807.3) == "-NT$807" and docs._nt(1234.5) == "NT$1,234"
    assert docs._rate(0.12) == "12%" and docs._rate(0.0725) == "7.25%" and docs._rate(0.105) == "10.5%"
    assert docs._date("2026-10-27") == "27 October 2026" and docs._month("2021-10-01") == "Oct 2021"
    assert docs._table(["a", "b"], [[1, 2]], "lr") == "| a | b |\n| --- | ---: |\n| 1 | 2 |"


def test_every_template_placeholder_has_a_value(values):
    for template in docs.outputs():
        text = (docs.TEMPLATE_DIR / template).read_text(encoding="utf-8")
        used = {m.group(1) for m in docs.PLACEHOLDER.finditer(text)}
        assert used <= set(values), (template, sorted(used - set(values)))


def test_documents_on_disk_are_what_the_templates_and_results_give(values):
    for template, target in docs.outputs().items():
        fresh = docs.render((docs.TEMPLATE_DIR / template).read_text(encoding="utf-8"), values)
        assert target.read_text(encoding="utf-8").rstrip("\n") == fresh.rstrip("\n"), f"{target.name} is out of date: run python -m creditrisk report"


def test_no_placeholder_or_missing_number_is_left_in_a_document():
    for target in docs.outputs().values():
        text = target.read_text(encoding="utf-8")
        assert "{{" not in text and "}}" not in text, target.name
        bad = r"\bnan\b|NaN|inf%|-inf|None%|NT\$None|: None\b|\(None\)"
        assert not re.search(bad, text), (target.name, re.findall(r".{20}(?:" + bad + r").{20}", text)[:3])


def test_every_relative_link_and_image_points_at_something_that_exists():
    for target in docs.outputs().values():
        for link in re.findall(r"\]\(([^)\s]+)\)", target.read_text(encoding="utf-8")):
            if link.startswith(("http", "#", "mailto")):
                continue
            path = (target.parent / link.split("#")[0]).resolve()
            assert path.exists(), f"{target.name} links to {link}"


def test_the_readme_quotes_the_results(values):
    text = (config.ROOT / "README.md").read_text(encoding="utf-8")
    s = values
    for needle in (f"{s['auc_scorecard']:.3f}", f"{s['auc_naive']:.3f}", f"{s['auc_challenger']:.3f}", s["break_even"], str(s["cutoff_score"]), s["profit_scorecard"], s["profit_all"], s["approved_share"]):
        assert needle in text, needle


def test_the_documents_carry_their_caveats():
    readme = (config.ROOT / "README.md").read_text(encoding="utf-8")
    assert "Taiwan" in readme and "not Pakistani borrowers" in readme and "assumption" in readme and "not lending advice" in readme
    assert "CC BY 4.0" in readme and "MIT" in readme
    memo = (config.DOCS / "recommendation_memo.md").read_text(encoding="utf-8")
    assert memo.startswith("# ") and "Recommendation" in memo and "What could be wrong" in memo and "Taiwanese" in memo
    card = (config.DOCS / "model_card.md").read_text(encoding="utf-8")
    assert all(h in card for h in ("## Intended use", "## Performance (hold-out)", "## Fairness", "## Limits and risks", "## Monitoring"))
    sources = (config.DOCS / "data_sources.md").read_text(encoding="utf-8")
    assert "Yeh" in sources and "CC BY 4.0" in sources and "What I could not get" in sources and config.MCB_CARD_URL in sources
    method = (config.DOCS / "methodology.md").read_text(encoding="utf-8")
    assert "## 9. The economics" in method and "break-even" in method


def test_the_gap_sentence_follows_the_confidence_interval(values):
    cmp = pd.read_csv(config.PROCESSED / "holdout_comparisons.csv")
    row = cmp[(cmp["model"] == "challenger") & (cmp["against"] == "scorecard") & (cmp["metric"] == "auc")].iloc[0]
    significant = row["low"] > 0 or row["high"] < 0
    assert ("larger than the noise" in values["gap_sentence"]) == significant and ("could be chance" in values["gap_sentence"]) == (not significant)


def test_the_fairness_headline_lists_exactly_the_flagged_groups(values):
    f = pd.read_csv(config.PROCESSED / "fairness_groups.csv")
    f = f[(f["model"] == "scorecard") & (f["accounts"] >= config.MIN_GROUP_SIZE)]
    names = {"sex": "sex", "age_band": "age", "education": "education", "marriage": "marital status"}
    flagged = [f"{names[r.attribute]}: {r.group}" for r in f.itertuples() if r.flagged]
    assert values["n_flagged"] == len(flagged)
    headline = values["fairness_headline"]
    if flagged:
        for name in flagged:
            assert name in headline
        for r in f.itertuples():
            if not r.flagged:
                assert f"{names[r.attribute]}: {r.group}" not in headline
    else:                                                                    # nobody flagged: it says so, and names the two groups closest to the line
        assert f"none of the {len(f)} groups" in headline
        for r in f.sort_values("disparate_impact").head(2).itertuples():
            assert f"{names[r.attribute]}: {r.group} ({r.disparate_impact:.2f})" in headline


def test_the_fairness_caveats_follow_the_flags_and_the_ranges(values):
    f = pd.read_csv(config.PROCESSED / "fairness_groups.csv")
    f = f[(f["model"] == "scorecard") & (f["accounts"] >= config.MIN_GROUP_SIZE)]
    names = {"sex": "sex", "age_band": "age", "education": "education", "marriage": "marital status"}
    if f["flagged"].any():
        assert "acceptable" in values["fairness_caveat"] and "acceptable" in values["fairness_close"]
    else:
        assert "does not make the gaps zero" in values["fairness_caveat"] and "clean bill of health" in values["fairness_close"]
        for r in f.itertuples():                                              # a group whose 95% range reaches under the line is named, and only such a group
            named = f"{names[r.attribute]}: {r.group}" in values["fairness_caveat"]
            assert named == (r.di_low < config.DISPARATE_IMPACT_MIN), (r.attribute, r.group)


def test_the_drivers_table_covers_the_flagged_and_the_close_groups_and_no_others(values):
    d = pd.read_csv(config.PROCESSED / "fairness_drivers.csv")
    shown = d["flagged"] | (d["disparate_impact"] < config.DISPARATE_IMPACT_WATCH)
    for (attribute, group), part in d.groupby(["attribute", "group"]):
        line = f"{docs.TITLES[attribute]}: {group} against "
        assert (line in values["drivers_table"]) == bool(shown[part.index].all()), (attribute, group)


def test_the_roll_rate_sentences_quote_the_results(values):
    split = pd.read_csv(config.PROCESSED / "rollrates_split.csv")
    base = split[split["write_off_status"] == config.ECON_WRITEOFF_STATUS].set_index("group")
    first, late = base.loc["Current in September"], base.loc["Late in September"]
    assert values["loss_current"] == docs._p(first["loss"], 1) and values["loss_late"] == docs._p(late["loss"], 1)
    assert values["loss_current"] in values["split_sentence"] and values["loss_late"] in values["split_sentence"]
    assert docs._p(first["share"], 0) in values["split_sentence"] and values["split_table"].count("\n") == 3          # header, rule and one line for each group
    mixed = float(first["share"] * first["loss"] + late["share"] * late["loss"])                           # the loss is linear in where the accounts start
    assert values["loss_base"] == docs._p(mixed, 1)


def test_the_explainable_model_sentence_quotes_the_check(values):
    check = pd.read_csv(config.PROCESSED / "monotone_check.csv")
    free = check[check["model"] == "challenger"]
    checks = (check["model"] == "monotone").sum() * int(check["accounts"].iloc[0])
    text = values["mono_sentence"]
    assert f"{int(free['violations'].sum()):,} of them" in text and f"({checks:,} account-and-feature checks)" in text
    assert f"{free['largest_fall'].max() * 100:.1f}" in text
    assert ("the monotone model in none" in text) == (check[check["model"] == "monotone"]["violations"].sum() == 0)


def test_the_sex_sentence_says_which_way_the_difference_goes(values):
    seg = pd.read_csv(config.PROCESSED / "segments.csv").query("variable == 'Sex'").set_index("group")["default_rate"]
    assert ("Women default less" in values["sex_sentence"]) == bool(seg["Female"] < seg["Male"])


def test_the_scorecard_profit_verdict_follows_the_range(values):
    s = pd.read_csv(config.PROCESSED / "strategies.csv").set_index("rule").loc["scorecard"]
    if s["low"] > 0:
        assert "clearly above zero" in values["scorecard_profit_verdict"]
    elif s["profit_per_account"] > 0:
        assert "not distinguishable from break-even" in values["scorecard_profit_verdict"]
    else:
        assert "does not earn a profit" in values["scorecard_profit_verdict"]


def test_scenario_sentences_name_the_right_dates(values):
    scen = pd.read_csv(config.PROCESSED / "scenarios.csv")
    peak = scen.loc[scen["policy_rate"].idxmax()]
    assert values["scen_peak_date"] == docs._date(peak["date"]) and values["scen_peak_rate"] == docs._rate(peak["policy_rate"])
    assert values["scen_first_date"] == docs._date(scen.iloc[0]["date"])


def test_the_example_rows_have_one_line_per_account(values):
    assert values["examples_rows"].count("\n") == 2 and values["examples_rows"].startswith("| Riskier than 90% of accounts")


def test_documents_are_written_where_the_readme_says(tmp_path):
    out = docs.outputs()
    assert out["readme.md"] == config.ROOT / "README.md" and {p.parent for k, p in out.items() if k != "readme.md"} == {Path(config.DOCS)}
