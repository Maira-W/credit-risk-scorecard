"""The figures. Each is drawn from the files in data/processed and saved to docs/figures; none holds a number typed by hand."""

from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from . import config, macro  # noqa: E402

NAVY, TEAL, ORANGE, AMBER, GREY, PURPLE = config.NAVY, config.TEAL, config.ORANGE, config.AMBER, config.GREY, config.PURPLE
MODEL_COLOURS = {"naive": GREY, "scorecard": NAVY, "challenger": TEAL, "scorecard_demographics": "#6C86A3", "challenger_demographics": "#7FC8BF"}
MODEL_LABELS = {"naive": "Naive: last month's status", "scorecard": "Scorecard", "challenger": "Gradient boosting",
                "scorecard_demographics": "Scorecard + demographics", "challenger_demographics": "Gradient boosting + demographics"}


def _style() -> None:
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E4E8EE",
                         "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white"})


def read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED / name, **kwargs)


def summary() -> dict:
    return json.loads((config.PROCESSED / "summary.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pct(ax, axis: str = "y", decimals: int = 0) -> None:
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=decimals))


# ---------------------------------------------------------------------------
def default_by_status() -> None:
    seg = read("segments.csv")
    s = seg[seg["variable"] == "Repayment status last month (September)"].sort_values("position")
    info = summary()
    base = info["data"]["default_rate"]
    strongest = max(info["scorecard"]["iv"], key=info["scorecard"]["iv"].get)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar([textwrap.fill(g, 11) for g in s["group"]], s["default_rate"], color=[TEAL if r < base else ORANGE for r in s["default_rate"]],
                  yerr=[s["default_rate"] - s["ci_low"], s["ci_high"] - s["default_rate"]], capsize=3, ecolor="#555")
    ax.axhline(base, color=NAVY, ls="--", lw=1)
    ax.text(-0.45, base + 0.012, f"all accounts {base:.1%}", ha="left", color=NAVY, fontsize=9)
    for bar, share in zip(bars, s["share"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.015, f"{share:.0%}\nof accounts", ha="center", va="bottom", color="white", fontsize=8)
    _pct(ax)
    ax.set_ylabel("share that defaulted the next month")
    ax.set_title("Last month's repayment status is the strongest single signal" if strongest == "status_sep" else "Default rate by last month's repayment status")
    ax.grid(axis="x", visible=False)
    save(fig, "01_default_by_status.png")


def risk_drivers() -> None:
    seg = read("segments.csv")
    base = summary()["data"]["default_rate"]
    blocks = ["Months late in the last six", "Share of the previous bill paid in September", "Balance as a share of the credit limit", "Credit limit"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, name in zip(axes.ravel(), blocks, strict=True):
        s = seg[seg["variable"] == name].sort_values("position")
        ax.bar(range(len(s)), s["default_rate"], color=NAVY, yerr=[s["default_rate"] - s["ci_low"], s["ci_high"] - s["default_rate"]], capsize=2, ecolor="#777")
        ax.axhline(base, color=ORANGE, ls="--", lw=1)
        ax.set_xticks(range(len(s)), [g.replace(" to ", "-").replace("NT$", "") for g in s["group"]], rotation=25, ha="right", fontsize=8)
        _pct(ax)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("default rate")
    axes[1, 0].set_ylabel("default rate")
    fig.suptitle("Default rate by months late, share of the last bill paid, credit used and credit limit (dashed: all accounts)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_risk_drivers.png")


def roc() -> None:
    curves, metrics = read("roc.csv"), read("holdout_metrics.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for model in ("naive", "scorecard", "challenger"):
        c = curves[curves["model"] == model]
        ax.plot(c["fpr"], c["tpr"], color=MODEL_COLOURS[model], lw=2, label=f"{MODEL_LABELS[model]} (AUC {metrics.loc[model, 'auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1, label="no skill (AUC 0.500)")
    ax.set_xlabel("share of good accounts declined (false positives)")
    ax.set_ylabel("share of defaulters declined (true positives)")
    ax.set_title("Ranking power on the hold-out accounts")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    save(fig, "03_roc.png")


def model_comparison() -> None:
    m = read("holdout_metrics.csv").set_index("model")
    cmp = read("holdout_comparisons.csv")
    demo = cmp[(cmp["metric"] == "auc") & cmp["model"].str.endswith("_demographics")]
    little = bool(((demo["difference"].abs() < 0.005) & (demo["low"] <= 0) & (demo["high"] >= 0)).all())
    order = ["naive", "scorecard", "scorecard_demographics", "challenger", "challenger_demographics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ("auc", "ks"), ("AUC (1.0 is perfect, 0.5 is a coin toss)", "KS (largest gap between good and bad)"), strict=True):
        y = np.arange(len(order))[::-1]
        ax.errorbar(m.loc[order, metric], y, xerr=[m.loc[order, metric] - m.loc[order, f"{metric}_low"], m.loc[order, f"{metric}_high"] - m.loc[order, metric]],
                    fmt="none", ecolor="#999", capsize=3)
        ax.scatter(m.loc[order, metric], y, c=[MODEL_COLOURS[k] for k in order], s=60, zorder=3)
        for yy, k in zip(y, order, strict=True):
            ax.text(m.loc[k, f"{metric}_high"] + 0.004, yy, f"{m.loc[k, metric]:.3f}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(np.arange(len(order))[::-1], [MODEL_LABELS[k] for k in order])
    fig.suptitle("Hold-out accuracy with 95% bootstrap ranges" + ("; adding sex, age, education and marital status changes little" if little else ""),
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_model_comparison.png")


def calibration() -> None:
    c = read("calibration.csv")
    fig, ax = plt.subplots(figsize=(6.2, 6))
    top = max(c["predicted"].max(), c["observed"].max()) * 1.05
    ax.plot([0, top], [0, top], color="#999", ls=":", lw=1, label="perfect")
    for model in ("scorecard", "challenger"):
        part = c[c["model"] == model]
        ax.plot(part["predicted"], part["observed"], "o-", color=MODEL_COLOURS[model], label=MODEL_LABELS[model])
    _pct(ax)
    _pct(ax, "x")
    ax.set_xlabel("predicted default rate (average in each tenth of accounts)")
    ax.set_ylabel("observed default rate")
    ax.set_title("Are the probabilities honest? (hold-out)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    save(fig, "05_calibration.png")


def lift() -> None:
    t = read("lift.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for model in ("naive", "scorecard", "challenger"):
        part = t[t["model"] == model]
        ax.plot([0, *(part["group"] / 10)], [0, *part["capture"]], "o-", color=MODEL_COLOURS[model], label=MODEL_LABELS[model], ms=4)
    ax.plot([0, 1], [0, 1], color="#999", ls=":", label="picking accounts at random")
    _pct(ax)
    _pct(ax, "x")
    ax.set_xlabel("riskiest accounts reviewed (share of all accounts)")
    ax.set_ylabel("share of all defaults found")
    ax.set_title("Finding the defaults early", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    s = t[t["model"] == "scorecard"]
    ax.bar(s["group"], s["default_rate"], color=NAVY)
    base = summary()["data"]["default_rate_holdout"]
    ax.axhline(base, color=ORANGE, ls="--", lw=1)
    for g, r in zip(s["group"], s["default_rate"], strict=True):
        ax.text(g, r + 0.01, f"{r:.0%}", ha="center", fontsize=8)
    _pct(ax)
    ax.set_xticks(range(1, 11), [f"{g}" for g in range(1, 11)])
    ax.set_xlabel("scorecard tenth (1 = riskiest)")
    ax.set_ylabel("default rate")
    ax.set_title("Default rate by scorecard tenth", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "06_lift_capture.png")


def score_distribution() -> None:
    h = read("score_distribution.csv")
    s = summary()
    cutoff = s["economics"]["cutoff_score"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = h["score_from"].diff().dropna().iloc[0] if len(h) > 1 else 10
    ax.bar(h["score_from"] + width / 2, h["good"], width=width * 0.92, color=TEAL, alpha=0.9, label="paid next month")
    ax.bar(h["score_from"] + width / 2, h["bad"], width=width * 0.92, bottom=h["good"], color=ORANGE, alpha=0.95, label="defaulted next month")
    if cutoff:
        ax.axvline(cutoff, color=NAVY, lw=1.5, ls="--", label=f"cut-off {cutoff}: approve this score and above")
    ax.set_xlabel("scorecard points (higher is safer)")
    ax.set_ylabel("accounts in the hold-out")
    ax.set_title(f"Scores separate the two groups (KS {s['metrics']['scorecard']['ks']:.2f})")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "07_score_distribution.png")


def scorecard_points() -> None:
    pts = read("scorecard_points.csv")
    ranges = pts.groupby("feature")["points"].agg(lambda v: v.max() - v.min()).sort_values(ascending=False)
    top = list(ranges.index[:4])
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    names = [config.feature_name(f) for f in ranges.index][::-1]
    ax.barh(names, ranges.values[::-1], color=NAVY)
    ax.set_xlabel("points between the best and worst bin")
    ax.set_title("What the card weighs", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    for k, feature in enumerate(top):
        ax2 = fig.add_subplot(grid[k // 2, 1 + k % 2])
        part = pts[pts["feature"] == feature]
        ax2.barh(range(len(part)), part["points"], color=[TEAL if p >= part["points"].median() else ORANGE for p in part["points"]])
        ax2.set_yticks(range(len(part)), part["bin"], fontsize=7)
        ax2.invert_yaxis()
        ax2.set_title(config.feature_name(feature), fontsize=9)
        ax2.set_xlim(min(0, part["points"].min() - 5), part["points"].max() + 8)
        for i, p in enumerate(part["points"]):
            ax2.text(p + 1, i, f"{p}", va="center", fontsize=7)
        ax2.grid(axis="y", visible=False)
    fig.suptitle("The scorecard: points for each bin of the four heaviest features (more points is safer)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_scorecard_points.png")


def importance() -> None:
    imp = read("challenger_importance.csv").head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh([config.feature_name(f) for f in imp["feature"]], imp["mean_abs_shap"], color=TEAL)
    ax.set_xlabel("average push on an account's log-odds of default (mean absolute SHAP value)")
    ax.set_title("What the gradient-boosting model relies on")
    ax.grid(axis="y", visible=False)
    save(fig, "09_importance.png")


def profit_curve() -> None:
    c = read("profit_curve.csv")
    s = summary()["economics"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model in ("scorecard", "challenger"):
        part = c[c["model"] == model]
        ax.plot(part["approved_share"], part["profit_per_account"], color=MODEL_COLOURS[model], lw=2, label=MODEL_LABELS[model])
    ax.axhline(0, color="#555", lw=1)
    ax.axvline(0, color="#555", lw=0.5)
    sc = c[c["model"] == "scorecard"]
    ok = sc[sc["cutoff"] <= s["break_even_pd"]]
    if len(ok):
        ax.axvline(ok["approved_share"].iloc[-1], color=NAVY, ls="--", lw=1)
        ax.text(ok["approved_share"].iloc[-1] + 0.01, ax.get_ylim()[0] * 0.2 if ax.get_ylim()[0] < 0 else 0, f"break-even probability {s['break_even_pd']:.1%}", color=NAVY, fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("share of accounts approved (safest first)")
    ax.set_ylabel("profit per account in the portfolio, NT$ a month")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Profit per account against the share of accounts approved")
    ax.legend(loc="lower left")
    save(fig, "10_profit_curve.png")


def policy_scenarios() -> None:
    sc, rate = read("scenarios.csv", parse_dates=["date"]), macro.policy_rate()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.step(rate["effective_date"], rate["policy_rate_pct"], where="post", color=NAVY, lw=2)
    ax.scatter(sc["date"], sc["policy_rate"] * 100, color=ORANGE, zorder=3, s=40)
    for _, r in sc.iterrows():
        ax.annotate(f"{r['policy_rate'] * 100:g}%", (r["date"], r["policy_rate"] * 100), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("policy rate, %")
    ax.set_title("SBP policy rate and the dates tested", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    labels = [d.strftime("%b %Y") for d in sc["date"]]
    ax = axes[1]
    ax.bar(labels, sc["approved_share"], color=NAVY)
    for i, v in enumerate(sc["approved_share"]):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    _pct(ax)
    ax.set_title("Share of accounts it pays to approve", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    x = np.arange(len(sc))
    ax.bar(x - 0.2, sc["profit_per_account"], 0.4, color=TEAL, label="cut-off reset to the rate")
    ax.bar(x + 0.2, sc["profit_per_account_first_cutoff_kept"], 0.4, color=ORANGE, label=f"{labels[0]} cut-off kept")
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x, labels, rotation=25)
    ax.set_title("Profit per account, NT$ a month", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.suptitle("The break-even cut-off and what it earns, on six policy-rate dates (card APR and the other assumptions held fixed)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "11_policy_rate_scenarios.png")


def sensitivity() -> None:
    s = read("sensitivity.csv")
    base = summary()["economics"]
    grid = s.pivot(index="apr", columns="loss", values="approved_share").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.6, float(grid.to_numpy().max())), aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [f"{v:.0%}" for v in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{v:.0%}" for v in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > 0.35 else "#222", fontsize=9,
                    fontweight="bold" if (abs(grid.index[i] - base["apr"]) < 1e-9 and abs(grid.columns[j] - base["loss"]) < 1e-9) else "normal")
    ax.set_xlabel("loss when an account defaults (share of its balance)")
    ax.set_ylabel("card APR")
    ax.set_title("Share of accounts worth approving, under different assumptions", fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, format=PercentFormatter(1.0), shrink=0.8)
    save(fig, "12_sensitivity.png")


def fairness() -> None:
    f = read("fairness_groups.csv")
    f = f[(f["model"] == "scorecard")]
    order = ["sex", "age_band", "education", "marriage"]
    titles = {"sex": "Sex", "age_band": "Age", "education": "Education", "marriage": "Marital status"}
    f = f.assign(attribute=pd.Categorical(f["attribute"], order)).sort_values(["attribute", "group"])
    labels = [f"{r.group}" for r in f.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(f))[::-1]
    judged = f["accounts"] >= config.MIN_GROUP_SIZE
    ax = axes[0]
    close = judged & ~f["flagged"] & (f["disparate_impact"] < config.DISPARATE_IMPACT_WATCH)                    # not flagged, but within 10 points of the line
    colours = [GREY if not j else (ORANGE if flag else (AMBER if c else NAVY)) for j, flag, c in zip(judged, f["flagged"], close, strict=True)]
    ax.barh(y, f["approval_rate"], color=colours)
    ax.set_yticks(y, [f"{titles[a]}: {g}" for a, g in zip(f["attribute"], labels, strict=True)], fontsize=8)
    for yy, v, ratio, j in zip(y, f["approval_rate"], f["disparate_impact"], judged, strict=True):
        ax.text(v + 0.005, yy, f"{v:.0%}" + (f"  (ratio {ratio:.2f})" if j else ""), va="center", fontsize=8)
    ax.set_xlim(0, f["approval_rate"].max() * 1.3)
    _pct(ax, "x")
    ax.set_title("Approval rate at the cut-off (orange: under the four-fifths line,\namber: within 10 points of it; ratio = rate over the best group's)", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(y + 0.18, f["default_rate"], 0.36, color=ORANGE, label="observed default rate")
    ax.barh(y - 0.18, f["mean_probability"], 0.36, color=NAVY, label="average predicted probability")
    ax.set_yticks(y, [""] * len(y))
    _pct(ax, "x")
    ax.set_title("Is the model honest for each group? (predicted against observed)", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Fairness audit of the scorecard's decisions on the hold-out accounts", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "13_fairness.png")


def pakistan() -> None:
    rate, npl, credit = macro.policy_rate(), pd.read_csv(config.PROCESSED / "pakistan_npl_ratio.csv"), pd.read_csv(config.PROCESSED / "pakistan_private_credit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(npl["year"], npl["value"], color=NAVY, lw=2)
    ax.set_ylabel("non-performing loans, % of gross loans")
    ax.set_title("Banks' non-performing loans (left) and the SBP policy rate (right)", fontsize=10)
    ax2 = ax.twinx()
    ax2.step(rate["effective_date"].dt.year + rate["effective_date"].dt.dayofyear / 365, rate["policy_rate_pct"], where="post", color=ORANGE, lw=1.6)
    ax2.set_ylabel("SBP policy rate, %", color=ORANGE)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax = axes[1]
    ax.plot(credit["year"], credit["value"], color=TEAL, lw=2)
    ax.set_ylabel("domestic credit to the private sector, % of GDP")
    ax.set_title("Domestic credit to the private sector", fontsize=10)
    fig.tight_layout()
    save(fig, "14_pakistan_context.png")


# ---------------------------------------------------------------------------
# The second layer
# ---------------------------------------------------------------------------
def rollrates() -> None:
    m, e = read("rollrates_matrix.csv"), read("rollrates_evidence.csv")
    names = list(dict.fromkeys(m["from_state"]))
    z = m.pivot(index="from_state", columns="to_state", values="probability").loc[names, names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.imshow(z.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    short = [n.replace("Late, status ", "Late ").replace(" or more", "+") for n in names]
    ax.set_xticks(range(len(names)), short, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), short, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = z.iloc[i, j]
            ax.text(j, i, f"{v:.0%}" if v >= 0.005 else "", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "#222")
    ax.set_xlabel("next month")
    ax.set_ylabel("this month")
    ax.set_title("Where accounts go next month (April to August)", fontsize=10)
    ax.grid(False)
    ax = axes[1]
    labels = [f"written off at\nstatus {int(r.write_off_status)}+\n(loss {r.loss:.1%})" for r in e.itertuples()]
    ax.bar(labels, e["probability"], color=[NAVY if s == config.ECON_WRITEOFF_STATUS else GREY for s in e["write_off_status"]],
           yerr=[e["probability"] - e["low"], e["high"] - e["probability"]], capsize=4, ecolor="#555")
    split = read("rollrates_split.csv")
    for group, colour, label in (("Late in September", ORANGE, "already late in September"), ("Current in September", TEAL, "first missed payment (current in September)")):
        part = split[split["group"] == group].sort_values("write_off_status")
        ax.scatter(range(len(part)), part["probability"], color=colour, s=50, zorder=4, label=label, edgecolor="white")
    top = max(e["high"].max(), split["probability"].max())
    ax.set_ylim(0, top * 1.3)                                                   # head room, so the legend stays clear of the bars
    ax.legend(fontsize=8, loc="upper right")
    _pct(ax)
    ax.set_ylabel(f"chance a missed payment is written off within {int(e['months'].iloc[0])} months")
    ax.set_title("What a missed payment costs (bars: all, with 95% range)", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "15_roll_rates.png")


def _case_label(text: str) -> str:
    """A loss case's name wrapped to fit under a bar: "Roll rates, written off at status 4 (base case)" becomes three short lines."""
    if text.startswith("Roll rates"):
        label = "Roll rates" + chr(10) + text.split("written off at ")[1].replace(" (base case)", "")
        return label + (chr(10) + "(base case)" if "base case" in text else "")
    return "Stress" + chr(10) + text.split(": ")[1].replace(" of the balance", "")


def loss_cases() -> None:
    c = read("loss_cases.csv")
    cases = list(dict.fromkeys(c["case"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    sc = c[c["rule"] == "scorecard"].set_index("case").loc[cases]
    ax.bar(range(len(cases)), sc["approved_share"], color=NAVY)
    for k, (_, r) in enumerate(sc.iterrows()):
        ax.text(k, r["approved_share"] + 0.015, f"{r['approved_share']:.0%}\n(under {r['break_even']:.0%})", ha="center", fontsize=8)
    ax.set_xticks(range(len(cases)), [_case_label(x) for x in cases], fontsize=8)
    _pct(ax)
    ax.set_ylim(0, 1.12)
    ax.set_title("Share of accounts worth approving, and the break-even probability", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    width = 0.26
    for offset, rule, colour, label in ((-width, "approve_all", GREY, "approve everyone"), (0, "scorecard", NAVY, "scorecard"), (width, "challenger", TEAL, "gradient boosting")):
        part = c[c["rule"] == rule].set_index("case").loc[cases]
        ax.bar(np.arange(len(cases)) + offset, part["profit_per_account"], width, color=colour, label=label,
               yerr=[part["profit_per_account"] - part["low"], part["high"] - part["profit_per_account"]], capsize=2, ecolor="#777")
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(range(len(cases)), [_case_label(x) for x in cases], fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("NT$ per account a month (95% range)")
    ax.set_title("What approving earns, under each loss", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "16_loss_cases.png")


def explainable_boosting() -> None:
    m, pr, v = read("advanced_models.csv").set_index("model"), read("advanced_profits.csv").set_index("rule"), read("monotone_check.csv")
    order = ["scorecard", "monotone", "challenger"]
    labels = {"scorecard": "Scorecard", "monotone": "Monotone boosting", "challenger": "Gradient boosting"}
    colours = {"scorecard": NAVY, "monotone": PURPLE, "challenger": TEAL}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    ax.errorbar(m.loc[order, "auc"], y, xerr=[m.loc[order, "auc"] - m.loc[order, "auc_low"], m.loc[order, "auc_high"] - m.loc[order, "auc"]], fmt="none", ecolor="#999", capsize=3)
    ax.scatter(m.loc[order, "auc"], y, c=[colours[k] for k in order], s=60, zorder=3)
    ax.set_yticks(y, [labels[k] for k in order])
    for yy, k in zip(y, order, strict=True):
        ax.text(m.loc[k, "auc_high"] + 0.003, yy, f"{m.loc[k, 'auc']:.3f}", va="center", fontsize=8)
    ax.set_title("AUC on the hold-out (95% range)", fontsize=10)
    ax = axes[1]
    rows = ["approve_all", *order]
    names = {"approve_all": "Approve everyone", **labels}
    y2 = np.arange(len(rows))[::-1]
    ax.errorbar(pr.loc[rows, "profit_per_account"], y2, xerr=[pr.loc[rows, "profit_per_account"] - pr.loc[rows, "low"], pr.loc[rows, "high"] - pr.loc[rows, "profit_per_account"]],
                fmt="none", ecolor="#999", capsize=3)
    ax.scatter(pr.loc[rows, "profit_per_account"], y2, c=[GREY if r == "approve_all" else colours[r] for r in rows], s=60, zorder=3)
    ax.axvline(0, color="#555", lw=1)
    ax.set_yticks(y2, [names[r] for r in rows])
    ax.set_title("Profit per account at the break-even cut-off, NT$ a month", fontsize=10)
    ax = axes[2]
    totals = v.groupby("model")[["violations", "accounts"]].sum()
    share = {name: totals.loc[name, "violations"] / totals.loc[name, "accounts"] for name in ("challenger", "monotone")}
    ax.bar(["Gradient\nboosting", "Monotone\nboosting"], [share["challenger"], share["monotone"]], color=[TEAL, PURPLE])
    for k, name in enumerate(("challenger", "monotone")):
        ax.text(k, share[name] + 0.02, f"{share[name]:.0%}\n({int(totals.loc[name, 'violations']):,} of {int(totals.loc[name, 'accounts']):,})", ha="center", va="bottom", fontsize=8)
    free = v[v["model"] == "challenger"]
    ax.text(0.5, 0.97, f"deepest fall of free boosting below an earlier\nlevel: {free['largest_fall'].max() * 100:.1f} percentage points", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#444")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _pct(ax)
    ax.set_title("Share of account-and-feature checks where the\nprobability moved the wrong way", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "17_explainable_boosting.png")


def swap_sets() -> None:
    t = read("swap_sets.csv")
    pair = "scorecard against gradient boosting"
    part = t[t["pair"] == pair].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colours = [NAVY, ORANGE, TEAL, GREY]
    labels = [c.replace(" approves", "\napproves") for c in part["cell"]]
    ax = axes[0]
    ax.bar(labels, part["share"], color=colours)
    for k, r in part.iterrows():
        ax.text(k, r["share"] + 0.012, f"{r['share']:.0%}\ndefaulted {r['default_rate']:.0%}", ha="center", fontsize=8)
    _pct(ax)
    ax.set_ylim(0, part["share"].max() * 1.25)
    ax.set_title("Who each model would approve", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    ax = axes[1]
    ax.bar(labels, part["profit_if_approved"], color=colours)
    ax.axhline(0, color="#555", lw=1)
    for k, r in part.iterrows():
        ax.text(k, r["profit_if_approved"], f"{r['profit_if_approved']:,.0f}", ha="center", va="bottom" if r["profit_if_approved"] >= 0 else "top", fontsize=8)
    ax.set_title("What approving them would earn, NT$ per account of the portfolio", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Swap sets: {pair}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "18_swap_sets.png")


def stress_and_pricing() -> None:
    st, pr = read("stress.csv"), read("pricing.csv")
    econ = summary()["economics"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.arange(len(st))
    ax.bar(x - 0.2, st["profit_kept"], 0.4, color=ORANGE, label="cut-off kept")
    ax.bar(x + 0.2, st["profit_reset"], 0.4, color=TEAL, label="cut-off re-set to the scenario")
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x, [textwrap.fill(n, 14) for n in st["scenario"]], fontsize=8)
    ax.set_ylabel("expected profit, NT$ per account a month")
    ax.set_title("Stress tests (expected values from the model's own probabilities)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    cap = 1.5
    shown = pr["required_apr"].clip(upper=cap)
    ax.bar(pr["group"], shown, color=[TEAL if ok else ORANGE for ok in pr["pays_at_current_apr"]])
    ax.axhline(econ["apr"], color=NAVY, ls="--", lw=1.2)
    ax.text(0.6, econ["apr"] + 0.03, f"card APR {econ['apr']:.0%}", color=NAVY, fontsize=9)
    for g, v, raw in zip(pr["group"], shown, pr["required_apr"], strict=True):
        if raw <= cap:
            ax.text(g, v + 0.02, f"{raw:.0%}", ha="center", fontsize=7, va="bottom")
        else:                                                                    # a bar cut off at the top of the scale: say so inside it, not above the axes
            ax.text(g, v - 0.04, f"{raw:.0%} (off the scale)", ha="center", fontsize=7, va="top", color="white", fontweight="bold", rotation=90)
    ax.set_ylim(0, cap + 0.12)
    ax.set_xticks(pr["group"])
    _pct(ax)
    ax.set_xlabel("scorecard tenth (1 = safest)")
    ax.set_ylabel("APR at which approving the tenth breaks even")
    ax.set_title("Risk-based pricing", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "19_stress_and_pricing.png")


def fairness_drivers() -> None:
    d = read("fairness_drivers.csv")
    watched = d[d["flagged"] | (d["disparate_impact"] < config.DISPARATE_IMPACT_WATCH)]            # flagged, or within 10 points of the four-fifths line
    groups = list(dict.fromkeys(zip(watched["attribute"], watched["group"], strict=True)))
    if not groups:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No group is flagged or close to the line at the base case", ha="center")
        ax.axis("off")
        save(fig, "20_fairness_drivers.png")
        return
    titles = {"sex": "Sex", "age_band": "Age", "education": "Education", "marriage": "Marital status"}
    features = (watched.groupby("feature")["points_gap"].apply(lambda v: v.abs().sum()).sort_values(ascending=False).index[:5]).tolist()
    palette = [NAVY, TEAL, ORANGE, AMBER, PURPLE, GREY]
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(groups) + 2.2))
    names = []
    for k, (attribute, group) in enumerate(groups):
        rows = d[(d["attribute"] == attribute) & (d["group"] == group)]
        part = rows.set_index("feature")["points_gap"]
        left_neg = left_pos = 0.0
        pieces = [(f, part.get(f, 0.0)) for f in features] + [("other", float(part.drop(features, errors="ignore").sum()))]
        for (name, value), colour in zip(pieces, palette, strict=True):
            label = config.feature_name(name) if name != "other" else "other features"
            if value < 0:
                ax.barh(k, value, left=left_neg, color=colour, label=label)
                left_neg += value
            else:
                ax.barh(k, value, left=left_pos, color=colour, label=label)
                left_pos += value
        ax.text(left_pos + 0.2, k, f"total {part.sum():+.1f}", va="center", fontsize=8)
        names.append(f"{titles[attribute]}: {group}\nagainst {rows['reference'].iloc[0]}")
    ax.set_yticks(range(len(groups)), names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#555", lw=1)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_xlabel("average scorecard points the group gets from each feature, against the best-treated group of the attribute")
    ax.set_title("What lies behind the approval gaps (groups flagged or within 10 points of the four-fifths line)", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "20_fairness_drivers.png")


def make_all() -> None:
    _style()
    for draw in (default_by_status, risk_drivers, roc, model_comparison, calibration, lift, score_distribution, scorecard_points, importance,
                 profit_curve, policy_scenarios, sensitivity, fairness, pakistan, rollrates, loss_cases, explainable_boosting, swap_sets,
                 stress_and_pricing, fairness_drivers):
        draw()
    print(f"[charts] {len(list(config.FIGURES.glob('*.png')))} figures in {config.FIGURES}")
