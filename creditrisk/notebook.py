"""Build and run the notebook that walks through the analysis.

`python -m creditrisk notebook` needs the extra tools in requirements-notebook.txt. The notebook refits the scorecard from the raw
file in a few seconds and checks the result against the committed hold-out numbers; the rest it reads from data/processed. It is
kept with its outputs so it can be read on GitHub without running anything; a test checks that its code is the code in this file
and that no cell failed.
"""

from __future__ import annotations

import json

from . import config

PATH = config.NOTEBOOKS / "credit_risk_walkthrough.ipynb"


def _pc(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


SETUP = '''import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
# the same functions the pipeline uses
from creditrisk import binning, build, config, economics, models, scorecard, validation  # noqa: E402

P = ROOT / "data" / "processed"
pd.options.display.float_format = "{:,.3f}".format
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})

summary = json.loads((P / "summary.json").read_text())
accounts, quality = build.clean(build.load_raw())
accounts = build.add_features(accounts)
print(f"{len(accounts):,} accounts, {accounts['default'].mean():.1%} defaulted the next month; {len(build.BEHAVIOUR)} behaviour columns")'''

QUALITY = '''# What cleaning found (the full record is data/processed/data_quality.json). Nothing was dropped.
pd.Series({k: v for k, v in quality.items() if not isinstance(v, dict)})'''

STATUS_QUIRK = '''# A repayment status of 1 ("one month late") hardly appears before September: the older months seem to be coded on another basis.
pd.DataFrame({"late (status 1 or more)": quality["late_share_by_month"], "status exactly 1": quality["status_one_by_month"],
              "status exactly 2": quality["status_two_by_month"]}).T[list(config.MONTHS)]'''

SEGMENTS = '''seg = pd.read_csv(P / "segments.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, variable in zip(axes, ["Repayment status last month (September)", "Credit limit"], strict=True):
    s = seg[seg["variable"] == variable].sort_values("position")
    ax.bar(range(len(s)), s["default_rate"], color="#1F3B57", yerr=[s["default_rate"] - s["ci_low"], s["ci_high"] - s["default_rate"]], capsize=2)
    ax.axhline(summary["data"]["default_rate"], color="#E76F51", ls="--")
    ax.set_xticks(range(len(s)), s["group"], rotation=25, ha="right", fontsize=8)
    ax.set_title(variable, fontsize=10)
    ax.set_ylabel("default rate")
plt.tight_layout()
plt.show()'''

SPLIT = '''# One split, made once: 80% to build everything on, 20% kept aside. Identical accounts never straddle it.
development, holdout = models.holdout_split(accounts)
dev, hold = accounts.iloc[development].reset_index(drop=True), accounts.iloc[holdout].reset_index(drop=True)
y_dev, y_hold = dev["default"].to_numpy(), hold["default"].to_numpy()
shared = set(dev["group"]) & set(hold["group"])
print(f"{len(dev):,} development and {len(hold):,} hold-out accounts; default rates {y_dev.mean():.2%} and {y_hold.mean():.2%}; "
      f"identical accounts on both sides: {len(shared)}")'''

WOE = '''# Weight of evidence by hand, for last month's repayment status.
b = binning.fit_binning(dev["status_sep"], y_dev, "status_sep", "numeric")
t = b.table
share_bad = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5 * len(t))      # half an account is added to every bin so that no bin has an infinite weight
share_good = (t["good"] + 0.5) / (t["good"].sum() + 0.5 * len(t))
by_hand = np.log(share_bad / share_good)
assert np.allclose(by_hand, t["woe"]) and np.isclose(((share_bad - share_good) * by_hand).sum(), b.iv)
print(f"information value {b.iv:.3f}: above 0.3 is strong")
t[["label", "n", "bad_rate", "woe", "iv"]]'''

FIT = '''# Fit the scorecard on the development accounts and compare with the committed hold-out result.
card = scorecard.Scorecard().fit(dev, y_dev)
p_hold = card.predict_proba(hold)
stored = summary["metrics"]["scorecard"]
print(f"hold-out AUC {validation.auc(y_hold, p_hold):.4f} here, {stored['auc']:.4f} in the results; KS {validation.ks(y_hold, p_hold):.4f} here, {stored['ks']:.4f} in the results")
assert abs(validation.auc(y_hold, p_hold) - stored["auc"]) < 1e-3 and abs(validation.ks(y_hold, p_hold) - stored["ks"]) < 2e-3
card.screening_[["feature", "iv", "decision", "reason"]].round(3)'''

POINTS = '''print(f"{len(card.features_)} features, {int(len(card.points_table()))} bins; the score is the sum of the points (higher is safer)")
card.points_range().rename("points between best and worst bin").to_frame().T'''

CARD = '''pts = card.points_table()
top = card.points_range().index[:3]
pts[pts["feature"].isin(top)].rename(columns={"default_rate": "default rate"}).round(3)'''

SCALING = '''s = card.scaling
odds = lambda score: (1 - s.probability(score)) / s.probability(score)  # noqa: E731
print(f"{s.base_score} points <-> probability of default {s.probability(s.base_score):.2%} (odds {s.base_odds}:1)")
print(f"{s.base_score + s.pdo} points: odds {odds(s.base_score + s.pdo):.1f}:1, exactly double; {s.base_score - s.pdo} points: odds {odds(s.base_score - s.pdo):.1f}:1")
card.reason_codes(hold.head(2), k=3)  # the reasons an account lost points, worst first'''

CHALLENGER = '''metrics = pd.read_csv(P / "holdout_metrics.csv").set_index("model")
metrics.loc[["naive", "scorecard", "challenger"], ["auc", "auc_low", "auc_high", "ks", "brier", "calibration_slope", "capture_top_decile"]]'''

SUBSETS = '''# The hard half: accounts that were not late last month, where a rule that only reads last month's status has nothing to go on.
pd.read_csv(P / "subset_metrics.csv")[["subset", "model", "accounts", "default_rate", "auc", "ks", "capture_top_decile"]]'''

CALIBRATION = '''cal = pd.read_csv(P / "calibration.csv")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 0.8], [0, 0.8], color="#999", ls=":")
for model, colour in [("scorecard", "#1F3B57"), ("challenger", "#2A9D8F")]:
    c = cal[cal["model"] == model]
    ax.plot(c["predicted"], c["observed"], "o-", color=colour, label=model)
ax.set_xlabel("predicted default rate")
ax.set_ylabel("observed default rate")
ax.legend()
plt.show()'''

BREAKEVEN = '''# The economics: an account that pays earns a margin on its balance, one that defaults loses a share of it.
e = summary["economics"]
econ = economics.Economics(apr=e["apr"], revolve_share=economics.revolve_share(dev), funding=e["funding"], opex=e["opex"], loss=e["loss"])
assert abs(econ.break_even - e["break_even_pd"]) < 1e-9
print(f"yield {econ.yield_:.1%} - funding {econ.funding:.2%} - running cost {econ.opex:.0%} = {econ.margin * 12:.1%} a year = {econ.margin:.3%} a month")
print(f"break-even probability of default = margin / (margin + loss) = {econ.break_even:.2%}; score {int(np.ceil(card.scaling.score(econ.break_even)))}")'''

CUTOFF = '''table = economics.threshold_table(y_hold, p_hold, economics.exposure(hold))
result = economics.profit_at(table, econ, econ.break_even)
row = pd.read_csv(P / "cutoffs.csv").query("model == 'scorecard' and rule == 'break-even probability'").iloc[0]
assert abs(result["approved_share"] - row["approved_share"]) < 1e-9
assert abs(result["realised_profit"] / len(hold) - row["profit_per_account"]) < 1e-6
print(f"approved {result['approved_share']:.1%} of hold-out accounts; {result['default_rate_approved']:.1%} of them defaulted; "
      f"profit {result['realised_profit'] / len(hold):,.0f} NT$ per account a month")
curve = economics.profit_curve(table, econ)
ax = (curve.set_index("approved_share")["realised_profit"] / len(hold)).plot(figsize=(8, 4))
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("share of accounts approved, safest first")
ax.set_ylabel("NT$ per account per month")
plt.show()'''

SCENARIOS = '''pd.read_csv(P / "scenarios.csv")[["date", "policy_rate", "break_even_pd", "approved_share", "profit_per_account", "profit_per_account_first_cutoff_kept"]]'''

FAIRNESS = '''f = pd.read_csv(P / "fairness_groups.csv").query("model == 'scorecard'")
f[["attribute", "group", "accounts", "default_rate", "mean_probability", "approval_rate", "disparate_impact", "flagged"]].round(3)'''


def cells() -> list[tuple[str, str]]:
    import pandas as pd
    s = json.loads((config.PROCESSED / "summary.json").read_text(encoding="utf-8"))
    m, e, d = s["metrics"], s["economics"], s["data"]
    share = float(pd.read_csv(config.PROCESSED / "strategies.csv").set_index("rule").loc["scorecard", "approved_share"])
    intro = f"""# Credit card default scorecard: a walkthrough

**Question.** Of {d['accounts']:,} credit card holders at a bank in Taiwan (April to September 2005), who will miss next month's payment, and where should a lender draw the line between approving and declining?

**Approach.** A points scorecard built the way banks build them (weight of evidence, logistic regression, points), tested against a gradient-boosting model and a naive rule, checked for calibration and fairness, then priced: the cut-off that earns most depends on the card's yield, the loss on a default, and the cost of money, which here is the State Bank of Pakistan's policy rate.

**Short answer.** On {d['holdout']:,} hold-out accounts the scorecard has an AUC of {m['scorecard']['auc']:.3f} against {m['naive']['auc']:.3f} for a rule that only reads last month's status and {m['challenger']['auc']:.3f} for gradient boosting. Under the stated assumptions an account is worth approving while its probability of default is under {_pc(e['break_even_pd'])}, which approves {_pc(share, 0)} of the hold-out accounts (section 8).

The cardholders are from Taiwan, not Pakistan, and "default" means a missed payment, not a write-off. The yield, running cost and loss in the economics are assumptions. This notebook refits the scorecard from the raw file in a few seconds and checks it against the committed results; everything else it reads from `data/processed`. Nothing here is lending advice."""
    limits = (f"## 11. Limits\n\n- **Not Pakistani data.** The accounts are Taiwanese from 2005, a crisis-era book where {_pc(d['default_rate'], 0)} missed a payment in one month. "
              "What carries over to Pakistan is the method and the link between the policy rate and the cut-off.\n"
              "- **Default is a missed payment**, not a charge-off, so the loss on a default is an assumption.\n"
              "- **One cohort.** There is no later period to test the scorecard's stability against.\n"
              "- **Selection.** Only accounts the bank had approved are here.\n"
              "- **The assumptions drive the cut-off.** Only the card APR and the policy rate are sourced; see the sensitivity table in the dashboard.")
    return [
        ("markdown", intro),
        ("code", SETUP),
        ("markdown", "## 1. What is in the file and what was cleaned\n\nThe file has no missing values. The documentation does not explain a few codes (education 0, 5 and 6, marital status 0), which are put with \"others\"; some identical accounts are mostly dormant cards. Nothing is dropped."),
        ("code", QUALITY),
        ("code", STATUS_QUIRK),
        ("markdown", "## 2. Who defaults\n\nLast month's repayment status is the strongest single signal, and the credit limit (which the bank itself chose) is a strong one too."),
        ("code", SEGMENTS),
        ("markdown", "## 3. The split\n\nEverything (bins, features, settings, cut-off) is chosen on the development accounts. The hold-out is used once, to report."),
        ("code", SPLIT),
        ("markdown", "## 4. Weight of evidence and information value\n\nA feature is cut into bins; each bin gets a weight of evidence, ln(share of bad accounts / share of good accounts), positive where the bin is riskier than average. The information value sums how far the bins pull good and bad apart. The cell computes both by hand for last month's status and checks them against the library."),
        ("code", WOE),
        ("markdown", "## 5. The scorecard\n\nThe full recipe is in `creditrisk/scorecard.py`: bin every candidate, drop features with little information, thin out correlated ones, fit a logistic regression and drop any feature that is not significant or has the wrong sign, then turn the regression into whole points. The cell below refits it on the development accounts and checks that it reproduces the committed hold-out result."),
        ("code", FIT),
        ("code", POINTS),
        ("code", CARD),
        ("markdown", "## 6. From points to probability\n\nThe score is 600 at odds of 20 good accounts to 1 bad one, and every 40 points double the odds. The probability of default is read straight from the whole-number score, so the card and the probability always agree."),
        ("code", SCALING),
        ("markdown", "## 7. How good is it?\n\nAgainst a naive rule and a gradient-boosting model, on the hold-out accounts. AUC is the chance that a random defaulter scores riskier than a random payer; KS is the largest gap between the two groups' cumulative score distributions. The second table looks only at accounts that were not late last month, where a rule that reads last month's status has nothing to go on."),
        ("code", CHALLENGER),
        ("code", SUBSETS),
        ("code", CALIBRATION),
        ("markdown", "## 8. Where to draw the line\n\nOver one billing month an approved account that pays earns a margin on its balance, and one that defaults loses a share of it. Approving pays while the probability of default is under margin / (margin + loss). The margin comes from the card's APR (MCB publishes 3.67% a month, up to 44% a year), the share of balances that revolve (measured in this file), the cost of money (the policy rate) and a running cost; the loss is an assumption."),
        ("code", BREAKEVEN),
        ("code", CUTOFF),
        ("markdown", "## 9. What the policy rate does to the cut-off\n\nWhen the State Bank raised its rate from 7.25% to 22%, the margin on a card balance shrank and the break-even probability fell: a lender that kept its 2021 cut-off through 2023 would have approved accounts that lost money."),
        ("code", SCENARIOS),
        ("markdown", "## 10. Fairness\n\nSex, age, education and marital status are not inputs. The audit shows what the decisions do to each group at the cut-off. The disparate-impact ratio flags a group approved at under four-fifths of the best group's rate; it is a screening rule, not a legal test in Pakistan."),
        ("code", FAIRNESS),
        ("markdown", limits),
    ]


def _merge_streams(notebook) -> None:
    """Join consecutive printed outputs of a cell into one. The kernel may deliver a line and its newline separately, which would
    make the saved file differ from run to run for no reason."""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        merged = []
        for output in cell.outputs:
            if merged and output.output_type == "stream" and merged[-1].output_type == "stream" and merged[-1].name == output.name:
                merged[-1].text += output.text
            else:
                merged.append(output)
        cell.outputs = merged


def build(execute: bool = True) -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    notebook_cells = []
    for number, (kind, source) in enumerate(cells(), 1):
        cell = new_code_cell(source) if kind == "code" else new_markdown_cell(source)
        cell["id"] = f"cell-{number:02d}"
        notebook_cells.append(cell)
    notebook = new_notebook(cells=notebook_cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                           "language_info": {"name": "python"}})
    if execute:
        from nbclient import NotebookClient
        config.NOTEBOOKS.mkdir(parents=True, exist_ok=True)
        NotebookClient(notebook, timeout=600, kernel_name="python3", record_timing=False,
                       resources={"metadata": {"path": str(config.NOTEBOOKS)}}).execute()
        _merge_streams(notebook)
    text = nbformat.writes(notebook)
    PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")      # Unix line endings on every machine


def run() -> None:
    build(execute=True)
    print(f"[notebook] wrote {PATH.relative_to(config.ROOT)}")
