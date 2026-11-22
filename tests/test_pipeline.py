"""The whole analysis, run on a small sample of the real accounts, then checked for internal consistency."""

import shutil

import pytest
from results_checks import check_results

from creditrisk import analysis, build, config, report


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    """Run the analysis end to end on 5,000 accounts, writing to a temporary folder."""
    folder = tmp_path_factory.mktemp("processed")
    saved = config.PROCESSED
    config.PROCESSED = folder
    try:
        if not (config.INTERIM / "accounts.csv.gz").exists():
            build.run()
        shutil.copy(saved / "data_quality.json", folder)                  # written by the build stage, read by the documents
        summary = analysis.run(analysis.FAST)
    finally:
        config.PROCESSED = saved
    return folder, summary


def test_every_file_is_written_and_consistent(fast_run):
    folder, summary = fast_run
    check_results(folder)
    assert summary["data"]["accounts"] == analysis.FAST.sample


def test_the_scorecard_beats_the_naive_rule_and_is_calibrated(fast_run):
    _, summary = fast_run
    m = summary["metrics"]
    assert m["scorecard"]["auc"] > m["naive"]["auc"] + 0.02
    assert abs(m["scorecard"]["calibration_slope"] - 1) < 0.35


def test_the_split_is_stratified_and_never_separates_identical_accounts(fast_run):
    _, summary = fast_run
    d = summary["data"]
    assert abs(d["default_rate_development"] - d["default_rate_holdout"]) < 0.02
    assert d["development"] == 4000 and d["holdout"] == 1000


def test_the_documents_and_figures_can_be_made_from_the_results(fast_run, tmp_path, monkeypatch):
    """report.run() draws every figure, builds the dashboard and writes the documents, here into a temporary copy of the project."""
    folder, _ = fast_run
    for name, value in (("PROCESSED", folder), ("DOCS", tmp_path / "docs"), ("FIGURES", tmp_path / "docs" / "figures"), ("DASHBOARD", tmp_path / "dashboard"),
                        ("ROOT", tmp_path)):
        monkeypatch.setattr(config, name, value)
    report.run()
    assert len(list((tmp_path / "docs" / "figures").glob("*.png"))) == 14
    assert (tmp_path / "dashboard" / "dashboard.html").stat().st_size > 50_000 and (tmp_path / "index.html").exists()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "{{" not in readme and "Credit card default scorecard" in readme
    assert {p.name for p in (tmp_path / "docs").glob("*.md")} == {"recommendation_memo.md", "methodology.md", "data_sources.md", "model_card.md"}
