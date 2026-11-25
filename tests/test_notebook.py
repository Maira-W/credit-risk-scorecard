"""The walkthrough notebook: its code is the code in notebook.py, every cell ran, and what it printed matches the results."""

import json

import pytest

from creditrisk import config, notebook

nbformat = pytest.importorskip("nbformat")


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(notebook.PATH, as_version=4)


def test_the_notebook_is_the_one_notebook_py_builds(nb):
    expected = notebook.cells()
    assert len(nb.cells) == len(expected)
    for cell, (kind, source) in zip(nb.cells, expected, strict=True):
        assert cell.cell_type == kind and cell.source == source


def test_every_code_cell_ran_without_an_error(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None, cell.source[:60]
            assert not [o for o in cell.outputs if o.output_type == "error"], cell.source[:60]


def test_the_notebook_validates_as_nbformat(nb):
    nbformat.validate(nb)


def test_the_refit_in_the_notebook_reproduced_the_committed_numbers(nb):
    summary = json.loads((config.PROCESSED / "summary.json").read_text(encoding="utf-8"))
    printed = "\n".join(o.get("text", "") for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs if o.output_type == "stream")
    assert f"{summary['metrics']['scorecard']['auc']:.4f} in the results" in printed
    assert f"{summary['data']['accounts']:,} accounts" in printed and f"{summary['economics']['break_even_pd']:.2%}" in printed


def test_the_notebook_has_the_plots_it_describes(nb):
    images = sum(1 for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs if o.output_type == "display_data" and "image/png" in o.data)
    assert images >= 3                      # the segments, the calibration plot and the profit curve
