"""Write README.md and the documents in docs/ from templates and the results, so a number in the text cannot drift from the data.
Edit the templates in creditrisk/templates/docs, not the output.

A placeholder is {{name}} or {{name:format}} (a Python format such as .1f or ,.0f). A name with no value is an error.
Every sentence that depends on what the data show is chosen here from the results, never typed in the template.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from . import config, macro
from .build import BEHAVIOUR, ENGINEERED

TEMPLATE_DIR = config.TEMPLATES / "docs"
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*(?::([^{}]+?))?\s*\}\}")


def outputs() -> dict:
    return {"readme.md": config.ROOT / "README.md", "memo.md": config.DOCS / "recommendation_memo.md", "methodology.md": config.DOCS / "methodology.md",
            "data_sources.md": config.DOCS / "data_sources.md", "model_card.md": config.DOCS / "model_card.md"}


def render(template: str, values: dict) -> str:
    def fill(m: re.Match) -> str:
        name, fmt = m.group(1), m.group(2)
        if name not in values:
            raise KeyError(f"template uses {name!r}, which has no value")
        return format(values[name], fmt) if fmt else str(values[name])
    return PLACEHOLDER.sub(fill, template)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list], align: str) -> str:
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _join(items) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _p(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _nt(x: float) -> str:
    return ("-" if x < 0 else "") + f"NT${abs(x):,.0f}"


def _f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _month(iso: str) -> str:
    return f"{pd.Timestamp(iso):%b %Y}"


def _date(iso: str) -> str:
    d = pd.Timestamp(iso)
    return f"{d.day} {d:%B %Y}"


def _rate(x: float) -> str:
    return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"


NAMES = {"naive": "Naive: last month's status only", "scorecard": "Scorecard", "scorecard_demographics": "Scorecard + sex, age, education, marriage",
         "challenger": "Gradient boosting", "challenger_demographics": "Gradient boosting + sex, age, education, marriage"}


# ---------------------------------------------------------------------------
# The results, loaded once
# ---------------------------------------------------------------------------
def load() -> dict:
    p = config.PROCESSED
    r = {"summary": json.loads((p / "summary.json").read_text(encoding="utf-8")),
         "quality": json.loads((p / "data_quality.json").read_text(encoding="utf-8")),
         "examples": json.loads((p / "examples.json").read_text(encoding="utf-8")),
         "manifest": json.loads(config.MANIFEST.read_text(encoding="utf-8")),
         "policy": macro.policy_rate()}
    for name in ("segments", "scorecard_points", "scorecard_screening", "scorecard_variants", "holdout_metrics", "holdout_comparisons", "cv_summary", "calibration",
                 "lift", "risk_bands", "stability_psi", "challenger_importance", "cutoffs", "strategies", "scenarios", "sensitivity", "fairness_groups",
                 "fairness_proxy", "subset_metrics", "challenger_tuning"):
        r[name] = pd.read_csv(p / f"{name}.csv")
    return r


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def metrics_table(r: dict) -> str:
    m, cv = r["holdout_metrics"].set_index("model"), r["cv_summary"].set_index("model")
    order = ["naive", "scorecard", "scorecard_demographics", "challenger", "challenger_demographics"]
    rows = [[NAMES[k], f"{cv.loc[k, 'auc_mean']:.3f}", f"**{m.loc[k, 'auc']:.3f}** ({m.loc[k, 'auc_low']:.3f} to {m.loc[k, 'auc_high']:.3f})",
             f"{m.loc[k, 'ks']:.3f} ({m.loc[k, 'ks_low']:.3f} to {m.loc[k, 'ks_high']:.3f})", f"{m.loc[k, 'brier']:.4f}", f"{m.loc[k, 'calibration_slope']:.2f}",
             _p(m.loc[k, "capture_top_decile"], 0)] for k in order]
    return _table(["Model", "AUC, cross-validated", "AUC, hold-out (95% range)", "KS, hold-out (95% range)", "Brier", "Calibration slope", "Defaults in riskiest tenth"],
                  rows, "lrrrrrr")


def comparisons_table(r: dict) -> str:
    c = r["holdout_comparisons"]
    rows = []
    for t in c[c["metric"] == "auc"].itertuples():
        verdict = "a real difference" if (t.low > 0 or t.high < 0) else "within the noise"
        rows.append([f"{NAMES[t.model]} against {NAMES[t.against].lower() if t.against == 'naive' else NAMES[t.against]}", f"{t.difference:+.4f}", f"{t.low:+.4f} to {t.high:+.4f}", verdict])
    return _table(["Comparison", "AUC difference", "95% range", "Verdict"], rows, "lrrl")


def segment_table(r: dict, variable: str) -> str:
    s = r["segments"]
    s = s[s["variable"] == variable].sort_values("position")
    return _table(["Group", "Accounts", "Share", "Default rate", "95% range"],
                  [[t.group, f"{t.accounts:,}", _p(t.share), _p(t.default_rate), f"{_p(t.ci_low)} to {_p(t.ci_high)}"] for t in s.itertuples()], "lrrrr")


def scorecard_table(r: dict, features: int | None = None) -> str:
    pts = r["scorecard_points"]
    order = list(dict.fromkeys(pts.assign(range=pts.groupby("feature")["points"].transform(lambda v: v.max() - v.min())).sort_values("range", ascending=False, kind="stable")["feature"]))
    rows = []
    for f in order[:features]:
        part = pts[pts["feature"] == f]
        for i, t in enumerate(part.itertuples()):
            rows.append([f"**{config.feature_name(f)}**" if i == 0 else "", t.bin, f"{t.accounts:,}", _p(t.default_rate), t.points])
    return _table(["Feature", "Bin", "Accounts", "Default rate", "Points"], rows, "llrrr")


def screening_table(r: dict) -> str:
    s = r["scorecard_screening"]
    rows = [[config.feature_name(t.feature), f"{t.iv:.3f}", t.decision, "" if pd.isna(t.reason) else t.reason] for t in s.itertuples()]
    return _table(["Candidate", "Information value", "Decision", "Why it was dropped"], rows, "lrll")


def strategies_table(r: dict) -> str:
    names = {"approve_all": "Approve everyone", "last_month_on_time": "Approve if not late last month (status 0 or better)", "scorecard": "Scorecard, at the break-even probability",
             "challenger": "Gradient boosting, at the break-even probability", "naive_model": "Naive model, at the break-even probability"}
    s = r["strategies"].set_index("rule")
    rows = []
    for k in ("approve_all", "last_month_on_time", "scorecard", "challenger"):
        t = s.loc[k]
        rows.append([names[k], _p(t.approved_share, 0), "" if pd.isna(t.default_rate_approved) else _p(t.default_rate_approved), _nt(t.profit_per_account),
                     f"{_nt(t.low)} to {_nt(t.high)}", "" if k == "approve_all" else f"{_nt(t.versus_approve_all)} ({_nt(t.versus_low)} to {_nt(t.versus_high)})"])
    return _table(["Rule", "Approved", "Default rate of approved", "Profit per account a month", "95% range", "Better than approving everyone by"], rows, "lrrrrr")


def scenarios_table(r: dict) -> str:
    s = r["scenarios"]
    rows = [[_date(t.date), _rate(t.policy_rate), _p(t.margin_month, 2), _p(t.break_even_pd), _p(t.approved_share, 0), _p(t.default_rate_approved), _nt(t.profit_per_account),
             _nt(t.profit_per_account_first_cutoff_kept)] for t in s.itertuples()]
    first = _month(s.iloc[0]["date"])
    return _table(["Policy rate on", "Rate", "Margin a month", "Break-even probability", "Approved", "Default rate of approved", "Profit per account a month",
                   f"If the {first} cut-off were kept"], rows, "lrrrrrrr")


def sensitivity_table(r: dict) -> str:
    s = r["sensitivity"]
    losses = sorted(s["loss"].unique())
    rows = []
    for apr in sorted(s["apr"].unique(), reverse=True):
        part = s[s["apr"] == apr].set_index("loss")
        rows.append([_p(apr, 0)] + [f"{_p(part.loc[x, 'approved_share'], 0)} (under {_p(part.loc[x, 'break_even_pd'])})" for x in losses])
    return _table(["Card APR, loss on a default:"] + [_p(x, 0) for x in losses], rows, "l" + "r" * len(losses))


def fairness_table(r: dict, model: str = "scorecard") -> str:
    f = r["fairness_groups"]
    f = f[f["model"] == model]
    titles = {"sex": "Sex", "age_band": "Age", "education": "Education", "marriage": "Marital status"}
    rows = []
    for t in f.itertuples():
        flag = "" if pd.isna(t.disparate_impact) else f"{t.disparate_impact:.2f}" + (" **flag**" if t.flagged else "")
        spread = "" if pd.isna(t.di_low) else f"{t.di_low:.2f} to {t.di_high:.2f}"
        rows.append([f"{titles[t.attribute]}: {t.group}", f"{t.accounts:,}", _p(t.default_rate), _p(t.mean_probability), f"{t.calibration_ratio:.2f}", _p(t.approval_rate, 0),
                     "" if pd.isna(t.default_rate_approved) else _p(t.default_rate_approved), flag, spread])
    return _table(["Group (hold-out)", "Accounts", "Default rate", "Predicted", "Predicted ÷ observed", "Approved", "Default rate of approved", "Disparate impact",
                   "95% range of the ratio"], rows, "lrrrrrrrr")


def importance_table(r: dict, n: int = 10) -> str:
    imp = r["challenger_importance"].head(n)

    def meaning(t) -> str:
        if t.feature.startswith("status_"):
            return "a code, not a quantity: late is riskier, but paying in full is not safer than paying the minimum"
        return "more default" if t.direction > 0.2 else "less default" if t.direction < -0.2 else "no clear direction"
    return _table(["Feature", "Share of the model's reliance", "A higher value means"], [[config.feature_name(t.feature), _p(t.share, 0), meaning(t)] for t in imp.itertuples()], "lrl")


def variants_rows(r: dict) -> dict:
    """The cross-validated AUC and number of features of each scorecard variant, keyed for the templates."""
    v = r["scorecard_variants"]
    out = {}
    for t in v.itertuples():
        key = "month_counts" if t.variant.startswith("With the month counts") else "limit_" + t.variant.split()[2].replace(".", "")
        out[f"var_{key}_auc"], out[f"var_{key}_features"] = f"{t.cv_auc:.3f}", f"{t.features:.1f}"
    return out


def subsets_table(r: dict) -> str:
    s = r["subset_metrics"]
    rows = [[t.subset, NAMES[t.model], f"{t.accounts:,}", _p(t.default_rate), f"{t.auc:.3f}", _p(t.capture_top_decile, 0)] for t in s.itertuples()]
    return _table(["Accounts", "Model", "Count", "Default rate", "AUC", "Defaults in riskiest tenth"], rows, "llrrrr")


def risk_bands_table(r: dict) -> str:
    return _table(["Band (scorecard probability)", "Accounts", "Default rate", "Share of all defaults"],
                  [[t.band, _p(t.share, 0), _p(t.default_rate), _p(t.share_of_defaults, 0)] for t in r["risk_bands"].itertuples()], "lrrr")


def _example_rows(r: dict) -> list[list]:
    rows = []
    for x in r["examples"]:
        why = "; ".join(f"{y['feature']}: {y['bin']} (-{y['points_lost']})" for y in x["reasons"]) or "none"
        rows.append([x["label"], x["score"], _p(x["probability"]), "yes" if x["defaulted"] else "no", why])
    return rows


def examples_table(r: dict) -> str:
    return _table(["Hold-out account", "Score", "Probability of default", "Defaulted?", "Reasons (points lost against the best bin)"], _example_rows(r), "lrrll")


def examples_rows(r: dict) -> str:
    """The same rows without a header, for a template that writes its own."""
    return "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in _example_rows(r))


def data_dictionary_table() -> str:
    meaning = {"limit": "Credit limit in NT$ (individual and family credit)", "sex_code": "1 male, 2 female", "education_code": "1 graduate school, 2 university, 3 high school, 4 other",
               "marriage_code": "1 married, 2 single, 3 other", "age": "Age in years", "default": "1 if the account missed its payment in October 2005"}
    rows = [["ID", "ID", "id", "Account number, 1 to 30,000"]]
    for x, uci in config.UCI_NAMES.items():
        mine = config.RENAME[x]
        if mine in meaning:
            text = meaning[mine]
        elif mine.startswith("status_"):
            text = f"Repayment status in {config.MONTH_NAMES[mine[7:]]} 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late"
        elif mine.startswith("bill_"):
            text = f"Bill statement in {config.MONTH_NAMES[mine[5:]]} 2005, NT$"
        else:
            text = f"Payment made in {config.MONTH_NAMES[mine[5:]]} 2005, NT$"
        rows.append([x, uci, mine, text])
    rows.append(["Y", "default payment next month", "default", meaning["default"]])
    return _table(["File column", "UCI name", "Name here", "Meaning"], rows, "llll")


def sources_table(r: dict) -> str:
    rows = [[f"`{name}`", e["what"], e["licence"], f"{e['bytes']:,}", e["retrieved"][:10], f"`{e['sha256'][:12]}...`"] for name, e in r["manifest"]["files"].items()]
    return _table(["File in data/raw", "What it is", "Licence", "Bytes", "Retrieved", "SHA-256"], rows, "llrrrl")


def policy_table(r: dict) -> str:
    return _table(["Effective", "Policy rate"], [[_date(str(t.effective_date.date())), f"{t.policy_rate_pct:g}%"] for t in r["policy"].itertuples()], "lr")


# ---------------------------------------------------------------------------
# The values every template may use
# ---------------------------------------------------------------------------
def values() -> dict:
    r = load()
    s, q = r["summary"], r["quality"]
    d, e, sc, ch, pk = s["data"], s["economics"], s["scorecard"], s["challenger"], s["pakistan"]
    m = r["holdout_metrics"].set_index("model")
    cmp = r["holdout_comparisons"]
    cv = r["cv_summary"].set_index("model")
    strat = r["strategies"].set_index("rule")
    cuts = r["cutoffs"]
    scen = r["scenarios"]
    sens = r["sensitivity"]
    fair = r["fairness_groups"]
    fair_sc = fair[fair["model"] == "scorecard"]
    subs = r["subset_metrics"].set_index(["subset", "model"])
    v: dict = {}

    def pair(model, against, metric="auc"):
        return cmp[(cmp["model"] == model) & (cmp["against"] == against) & (cmp["metric"] == metric)].iloc[0]

    # --- data
    v.update(accounts=d["accounts"], defaults=d["defaults"], default_rate=d["default_rate"], default_rate_pct=_p(d["default_rate"]), n_dev=d["development"], n_hold=d["holdout"],
             hold_defaults=round(d["default_rate_holdout"] * d["holdout"]), default_rate_hold_pct=_p(d["default_rate_holdout"]),
             education_recoded=q["education_recoded"], marriage_recoded=q["marriage_recoded"], repeated_rows=q["rows_in_repeated_groups"], extra_copies=q["extra_copies"],
             repeated_dormant=q["repeated_rows_dormant"], negative_bill_accounts=q["accounts_with_a_negative_bill"], over_limit=q["over_limit_in_september"],
             no_bills=q["no_bills_in_six_months"], no_payments=q["no_payments_in_six_months"],
             education_other_total=q["education_codes_before"]["4"] + q["education_recoded"], marriage_other_total=q["marriage_codes_before"]["3"] + q["marriage_recoded"],
             late_sep_pct=_p(q["late_share_by_month"]["sep"]), late_aug_pct=_p(q["late_share_by_month"]["aug"]), late_apr_pct=_p(q["late_share_by_month"]["apr"]),
             status_one_sep=q["status_one_by_month"]["sep"], status_one_aug=q["status_one_by_month"]["aug"], status_one_rest=sum(q["status_one_by_month"][k] for k in ("jul", "jun", "may", "apr")),
             status_two_aug=q["status_two_by_month"]["aug"], n_columns=len(BEHAVIOUR), n_engineered=len(ENGINEERED))
    seg = r["segments"]
    status = seg[seg["variable"] == "Repayment status last month (September)"].set_index("group")
    v.update(rate_status_ok=_p(status.loc["Paid the minimum", "default_rate"]), rate_status_one=_p(status.loc["1 month late", "default_rate"], 0),
             rate_status_two=_p(status.loc["2 months late", "default_rate"], 0), rate_status_three=_p(status.loc["3 or more months late", "default_rate"], 0))
    sex = seg[seg["variable"] == "Sex"].set_index("group")
    v.update(rate_female=_p(sex.loc["Female", "default_rate"]), rate_male=_p(sex.loc["Male", "default_rate"]))
    limit = seg[seg["variable"] == "Credit limit"].sort_values("position")
    v.update(rate_limit_low=_p(limit.iloc[0]["default_rate"], 0), rate_limit_high=_p(limit.iloc[-1]["default_rate"], 0))
    late_share = status.loc[["1 month late", "2 months late", "3 or more months late"], "share"].sum()
    defaults_late = (status.loc[["1 month late", "2 months late", "3 or more months late"], "defaults"].sum()) / d["defaults"]
    v.update(late_share=_p(late_share, 0), defaults_late_share=_p(defaults_late, 0), defaults_not_late_share=_p(1 - defaults_late, 0))

    # --- models
    sa, ca, na = m.loc["scorecard"], m.loc["challenger"], m.loc["naive"]
    gap = pair("challenger", "scorecard")
    demo_s, demo_c = pair("scorecard_demographics", "scorecard"), pair("challenger_demographics", "challenger")
    over_naive = pair("scorecard", "naive")
    v.update(auc_scorecard=sa["auc"], auc_scorecard_lo=sa["auc_low"], auc_scorecard_hi=sa["auc_high"], auc_naive=na["auc"], auc_challenger=ca["auc"], auc_challenger_lo=ca["auc_low"],
             auc_challenger_hi=ca["auc_high"], ks_scorecard=sa["ks"], ks_challenger=ca["ks"], ks_naive=na["ks"], gini_scorecard=sa["gini"],
             brier_scorecard=sa["brier"], slope_scorecard=sa["calibration_slope"], slope_challenger=ca["calibration_slope"], slope_naive=na["calibration_slope"],
             cv_auc_scorecard=cv.loc["scorecard", "auc_mean"], cv_auc_scorecard_sd=cv.loc["scorecard", "auc_std"], cv_auc_challenger=cv.loc["challenger", "auc_mean"],
             cv_auc_naive=cv.loc["naive", "auc_mean"], capture_scorecard=_p(sa["capture_top_decile"], 0), capture_naive=_p(na["capture_top_decile"], 0),
             capture_challenger=_p(ca["capture_top_decile"], 0), capture3_scorecard=_p(sa["capture_top_3_deciles"], 0), capture3_naive=_p(na["capture_top_3_deciles"], 0),
             capture3_challenger=_p(ca["capture_top_3_deciles"], 0), auc_gap=gap["difference"], auc_gap_lo=gap["low"], auc_gap_hi=gap["high"],
             auc_over_naive=over_naive["difference"], share_of_lift=_p((sa["auc"] - 0.5) / (ca["auc"] - 0.5), 0), logloss_before=ch["log_loss_before_recalibration"],
             logloss_after=ch["log_loss_after_recalibration"], platt_slope=ch["platt_slope"], platt_intercept=ch["platt_intercept"],
             rounding_change=sc["rounding_auc_change"], bootstrap_draws=s["settings"]["bootstrap_draws"], cv_folds=s["settings"]["cv_folds"], cv_repeats=s["settings"]["cv_repeats"],
             tuning_trials=s["settings"]["tuning_trials"])
    sig_gap = gap["low"] > 0 or gap["high"] < 0
    v["gap_sentence"] = (f"Gradient boosting ranks better by {gap['difference']:.3f} AUC (95% range {gap['low']:.3f} to {gap['high']:.3f}), a difference that is larger than the noise of a "
                         f"{d['holdout']:,}-account test" if sig_gap else
                         f"Gradient boosting ranks better by {gap['difference']:.3f} AUC (95% range {gap['low']:.3f} to {gap['high']:.3f}), a difference that could be chance on a "
                         f"{d['holdout']:,}-account test")
    v["gap_verdict"] = "real but small" if sig_gap and gap["difference"] < 0.03 else "real" if sig_gap else "within the noise"
    lift_share = (sa["auc"] - 0.5) / (ca["auc"] - 0.5)
    v["h_compare"] = ("A transparent scorecard gets most of the way to a machine-learning model." if lift_share >= 0.85
                      else "A machine-learning model ranks clearly better than the transparent scorecard.")
    v["h_calibration"] = ("Its probabilities are honest." if abs(sa["calibration_slope"] - 1) < 0.1 else "Its probabilities are not well calibrated.")
    v["demographics_sentence"] = (f"Adding sex, age, education and marital status changes the scorecard's hold-out AUC by {demo_s['difference']:+.4f} (95% range {demo_s['low']:+.4f} to {demo_s['high']:+.4f}) "
                                  f"and the gradient-boosting model's by {demo_c['difference']:+.4f} ({demo_c['low']:+.4f} to {demo_c['high']:+.4f}), so leaving them out costs "
                                  + ("almost nothing" if max(abs(demo_s["difference"]), abs(demo_c["difference"])) < 0.005 else "a little accuracy"))
    v["demo_cost"] = "almost no accuracy" if max(abs(demo_s["difference"]), abs(demo_c["difference"])) < 0.005 else "a little accuracy"
    hard_n, hard_s, hard_c = subs.loc[("Not late in September", "naive")], subs.loc[("Not late in September", "scorecard")], subs.loc[("Not late in September", "challenger")]
    v["h_hard"] = ("It finds defaults the naive rule cannot." if hard_s["auc"] > hard_n["auc"] + 0.05
                   else "It adds little over the naive rule among accounts that were not late.")
    v.update(hard_accounts=int(hard_n["accounts"]), hard_default_rate=_p(hard_n["default_rate"]), hard_auc_naive=hard_n["auc"], hard_auc_scorecard=hard_s["auc"], hard_auc_challenger=hard_c["auc"],
             hard_defaults=int(hard_n["defaults"]))

    # --- the scorecard
    top_features = list(sc["points_range"])[:3]
    v.update(n_features=sc["n_features"], n_bins=sc["n_bins"], n_candidates=sc["candidates"], base_score=sc["base_score"], base_odds=sc["base_odds"], pdo=sc["pdo"], score_min=sc["score_min"],
             score_max=sc["score_max"], top_feature_1=config.feature_name(top_features[0]), top_feature_2=config.feature_name(top_features[1]), top_feature_3=config.feature_name(top_features[2]),
             top_range_1=sc["points_range"][top_features[0]], top_range_2=sc["points_range"][top_features[1]], top_range_3=sc["points_range"][top_features[2]],
             features_list=_join(config.feature_name(f) for f in sc["features"]), n_dropped=sc["candidates"] - sc["n_features"], psi_score=s["psi"]["score"], psi_max=s["psi"]["max_feature"])
    screening = r["scorecard_screening"]
    cal_sc = r["calibration"][r["calibration"]["model"] == "scorecard"]
    v["cal_gap_pp"] = f"{(cal_sc['predicted'] - cal_sc['observed']).abs().max() * 100:.1f}"
    v["cv_hold_phrase"] = ("so the hold-out is typical" if abs(sa["auc"] - cv.loc["scorecard", "auc_mean"]) <= 2 * cv.loc["scorecard", "auc_std"]
                           else "which differs from the hold-out by more than two standard deviations")
    v["hard_phrase"] = ("; accounts that have not yet shown trouble leave less to see" if hard_s["auc"] < sa["auc"] else "")
    v.update(variants_rows(r), base_odds_pd=_p(1 / (1 + sc["base_odds"])), min_group=config.MIN_GROUP_SIZE, examples_rows=examples_rows(r))
    variants = r["scorecard_variants"].set_index("variant")
    pick = lambda text: variants[variants.index.str.contains(text, regex=False)].iloc[0]  # noqa: E731
    gap_auc, fewer = float(pick("0.7")["cv_auc"] - pick("(used)")["cv_auc"]), float(pick("0.7")["features"] - pick("(used)")["features"])
    v["limit_choice"] = ("The limit of 0.6 was chosen: against 0.7 it " + (f"gives up {gap_auc:.3f} of cross-validated AUC" if gap_auc > 0 else "loses no cross-validated AUC")
                         + f", for a card with {abs(fewer):.1f} {'fewer' if fewer >= 0 else 'more'} features on average")
    gain = float(variants[variants.index.str.startswith("With the month counts")]["cv_auc"].iloc[0] - variants[variants.index.str.contains(r"\(used\)")]["cv_auc"].iloc[0])
    v["platt_sentence"] = (
        f"Its probabilities were already close to calibrated, so a logistic correction (slope {ch['platt_slope']:.2f}, intercept {ch['platt_intercept']:.2f}), fitted on out-of-fold predictions "
        f"of the development accounts, changes almost nothing: the hold-out log loss moves from {ch['log_loss_before_recalibration']:.4f} to {ch['log_loss_after_recalibration']:.4f}."
        if abs(ch["platt_slope"] - 1) < 0.03 and abs(ch["platt_intercept"]) < 0.05 else
        f"Its probabilities are {'slightly too extreme' if ch['platt_slope'] < 1 else 'slightly too timid'} out of the box, so a logistic correction (slope {ch['platt_slope']:.2f}, intercept "
        f"{ch['platt_intercept']:.2f}), fitted on out-of-fold predictions of the development accounts, adjusts them; on the hold-out the log loss moves from "
        f"{ch['log_loss_before_recalibration']:.4f} to {ch['log_loss_after_recalibration']:.4f}.")
    v["month_counts_gain"] = gain
    v["month_counts_verdict"] = "a gain too small to be worth a reason code nobody could explain" if gain < 0.005 else "a real gain, given up for explainability"
    v["n_dropped_correlated"] = int(screening["reason"].fillna("").str.startswith("correlated").sum())
    v["n_dropped_regression"] = int(screening["reason"].fillna("").str.startswith(("coefficient", "p-value")).sum())
    v["n_dropped_iv"] = int(screening["reason"].fillna("").str.startswith(("information value", "a single bin")).sum())
    imp = r["challenger_importance"]
    v["challenger_top"] = _join(config.feature_name(f) for f in imp["feature"].head(3))

    # --- the economics
    sc_row, ch_row, all_row, ok_row = strat.loc["scorecard"], strat.loc["challenger"], strat.loc["approve_all"], strat.loc["last_month_on_time"]
    emp = cuts[(cuts["model"] == "scorecard") & (cuts["rule"] == "best on the development accounts")].iloc[0]
    v["emp_phrase"] = "close to" if abs(emp["cutoff"] - e["break_even_pd"]) < 0.02 else "different from"
    v.update(apr=_p(e["apr"], 0), revolve_pct=_p(e["revolve_share"], 0), yield_pct=_p(e["yield"]), funding_pct=_rate(e["funding"]), opex_pct=_p(e["opex"], 0), loss_pct=_p(e["loss"], 0),
             margin_month_pct=_p(e["margin_month"], 2), margin_year_pct=_p(e["margin_month"] * 12), break_even=_p(e["break_even_pd"]), break_even_value=e["break_even_pd"],
             cost_ratio=e["cost_ratio"], cutoff_score=e["cutoff_score"], policy_date=_date(e["policy_rate_date"]), mean_balance=_nt(e["mean_balance"]),
             approved_share=_p(sc_row["approved_share"], 0), approved_share_precise=_p(sc_row["approved_share"]), bad_rate_approved=_p(sc_row["default_rate_approved"]),
             profit_scorecard=_nt(sc_row["profit_per_account"]), profit_scorecard_lo=_nt(sc_row["low"]), profit_scorecard_hi=_nt(sc_row["high"]), profit_all=_nt(all_row["profit_per_account"]),
             profit_all_lo=_nt(all_row["low"]), profit_all_hi=_nt(all_row["high"]), gain_scorecard=_nt(sc_row["versus_approve_all"]), gain_scorecard_lo=_nt(sc_row["versus_low"]),
             gain_scorecard_hi=_nt(sc_row["versus_high"]), profit_challenger=_nt(ch_row["profit_per_account"]), approved_challenger=_p(ch_row["approved_share"], 0),
             profit_ontime=_nt(ok_row["profit_per_account"]), approved_ontime=_p(ok_row["approved_share"], 0), bad_rate_ontime=_p(ok_row["default_rate_approved"]),
             empirical_cutoff=emp["cutoff"], empirical_cutoff_pct=_p(emp["cutoff"]), empirical_share=_p(emp["approved_share"], 0), empirical_profit=_nt(emp["profit_per_account"]))
    gap_row = strat.loc["challenger"]
    v["ch_vs_sc_phrase"] = f"{_nt(abs(gap_row['versus_scorecard']))} {'more' if gap_row['versus_scorecard'] >= 0 else 'less'} per account a month than the scorecard"
    v.update(ch_vs_sc_lo=_nt(gap_row["versus_scorecard_low"]), ch_vs_sc_hi=_nt(gap_row["versus_scorecard_high"]))
    v["ch_vs_sc_verdict"] = ("a difference larger than the noise" if gap_row["versus_scorecard_low"] > 0 or gap_row["versus_scorecard_high"] < 0
                             else "a difference that could be chance on a test this size")
    v["scorecard_profit_verdict"] = ("The scorecard's profit is clearly above zero on this test" if sc_row["low"] > 0 else
                                     f"The scorecard's profit is positive on average, but its 95% range reaches below zero, so it is not distinguishable from break-even on {d['holdout']:,} accounts"
                                     if sc_row["profit_per_account"] > 0 else "The scorecard does not earn a profit under these assumptions")
    v["all_phrase"] = f"Approving everyone {'loses' if all_row['profit_per_account'] < 0 else 'earns'} {_nt(abs(all_row['profit_per_account']))} per account a month in this book"
    v["gain_verdict"] = ("and the gain over approving everyone is larger than the noise" if sc_row["versus_low"] > 0 else "but the gain over approving everyone is not clearly larger than the noise")

    # --- scenarios
    first, last, peak = scen.iloc[0], scen.iloc[-1], scen.loc[scen["policy_rate"].idxmax()]
    v.update(scen_first_date=_date(first["date"]), scen_first_rate=_rate(first["policy_rate"]), scen_first_be=_p(first["break_even_pd"]), scen_first_share=_p(first["approved_share"], 0),
             scen_peak_date=_date(peak["date"]), scen_peak_rate=_rate(peak["policy_rate"]), scen_peak_be=_p(peak["break_even_pd"]), scen_peak_share=_p(peak["approved_share"], 0),
             scen_peak_profit=_nt(peak["profit_per_account"]), scen_peak_kept=_nt(peak["profit_per_account_first_cutoff_kept"]),
             scen_peak_kept_share=_p(peak["approved_share_first_cutoff_kept"], 0), scen_last_date=_date(last["date"]), scen_last_rate=_rate(last["policy_rate"]),
             scen_last_be=_p(last["break_even_pd"]), scen_last_share=_p(last["approved_share"], 0))
    v["scen_peak_verdict"] = ("keeping the earlier cut-off would have lost money on the accounts it approved" if peak["profit_per_account_first_cutoff_kept"] < 0 else
                              "keeping the earlier cut-off would have earned less than resetting it")
    v.update(sens_be_min=_p(sens["break_even_pd"].min()), sens_be_max=_p(sens["break_even_pd"].max()), sens_share_min=_p(sens["approved_share"].min(), 0),
             sens_share_max=_p(sens["approved_share"].max(), 0))

    # --- fairness
    judged = fair_sc[fair_sc["accounts"] >= config.MIN_GROUP_SIZE]
    label = {"sex": "sex", "age_band": "age", "education": "education", "marriage": "marital status"}
    flagged_rows = [t for t in judged.itertuples() if t.flagged]
    flagged = [f"{label[t.attribute]}: {t.group}" for t in flagged_rows]
    sure = [f"{label[t.attribute]}: {t.group}" for t in flagged_rows if t.di_high < config.DISPARATE_IMPACT_MIN]
    v["flagged_groups"] = _join(flagged)
    v["n_flagged"] = len(flagged)
    if flagged:
        v["fairness_headline"] = (f"At the break-even cut-off, {len(flagged)} of {len(judged)} groups are approved at under four-fifths of the best-treated group's rate: {_join(flagged)}"
                                  + (f"; for {_join(sure)} the whole 95% bootstrap range of the ratio is under 0.80, for the rest it reaches above it." if sure else
                                     "; for none of them is the whole 95% bootstrap range of the ratio under 0.80, so each could be chance on a test this size."))
    else:
        v["fairness_headline"] = f"At the break-even cut-off, none of the {len(judged)} groups is approved at under four-fifths of the best-treated group's rate."
    cal = (judged["calibration_ratio"] - 1).abs()
    v.update(worst_calibration=_p(cal.max(), 0), worst_calibration_group=f"{label[judged.loc[cal.idxmax(), 'attribute']]}: {judged.loc[cal.idxmax(), 'group']}",
             min_di=f"{judged['disparate_impact'].min():.2f}", calibration_min=f"{judged['calibration_ratio'].min():.2f}", calibration_max=f"{judged['calibration_ratio'].max():.2f}")
    sexes = fair_sc[fair_sc["attribute"] == "sex"].set_index("group")
    v.update(approval_female=_p(sexes.loc["Female", "approval_rate"], 0), approval_male=_p(sexes.loc["Male", "approval_rate"], 0), di_male=f"{sexes.loc['Male', 'disparate_impact']:.2f}",
             di_female=f"{sexes.loc['Female', 'disparate_impact']:.2f}", cal_female=f"{sexes.loc['Female', 'calibration_ratio']:.2f}", cal_male=f"{sexes.loc['Male', 'calibration_ratio']:.2f}")
    lower_female = sexes.loc["Female", "default_rate"] < sexes.loc["Male", "default_rate"]
    more_female = sexes.loc["Female", "approval_rate"] > sexes.loc["Male", "approval_rate"]
    v["sex_sentence"] = (f"Women default less than men in this file ({v['rate_female']} against {v['rate_male']}) and are approved {'more' if more_female else 'less'} often "
                         f"({v['approval_female']} against {v['approval_male']}); the model is calibrated for both (predicted ÷ observed {v['cal_female']} for women, {v['cal_male']} for men)."
                         if lower_female else
                         f"Women default more than men in this file ({v['rate_female']} against {v['rate_male']}) and are approved {'more' if more_female else 'less'} often "
                         f"({v['approval_female']} against {v['approval_male']}); predicted ÷ observed is {v['cal_female']} for women and {v['cal_male']} for men.")
    if flagged_rows:
        higher, ratios = [], []
        for t in flagged_rows:
            peers = judged[judged["attribute"] == t.attribute]
            best_treated = peers.loc[peers["approval_rate"].idxmax()]
            higher.append(t.default_rate > best_treated["default_rate"])
            ratios.append(t.calibration_ratio)
        v["fairness_reading"] = (
            f"The flagged groups have higher default rates in this file than the best-treated group of the same attribute, and the model is calibrated for them (predicted ÷ observed "
            f"between {min(ratios):.2f} and {max(ratios):.2f}), so the gaps in approval mostly follow gaps in observed default rates, not a model that is harsher than the data on those groups."
            if all(higher) and min(ratios) > 0.85 and max(ratios) < 1.15 else
            f"The gaps do not all follow gaps in observed default rates and calibration (predicted ÷ observed for the flagged groups runs from {min(ratios):.2f} to {max(ratios):.2f}): see the table.")
    else:
        v["fairness_reading"] = "No group is flagged."
    proxy = r["fairness_proxy"].set_index("attribute")["auc"]
    v.update(proxy_sex=proxy["Sex (female)"], proxy_age=proxy["Age 40 or over"], proxy_edu=proxy["University or higher"])
    ages = fair_sc[fair_sc["attribute"] == "age_band"]
    young = ages.iloc[0]
    v.update(age_band_young=young["group"], age_young_default=_p(young["default_rate"]), age_young_approval=_p(young["approval_rate"], 0), age_young_di=f"{young['disparate_impact']:.2f}")

    # --- Pakistan
    v.update(rate_now=f"{pk['rate_now']:g}%", rate_now_date=_date(pk["rate_now_date"]), rate_peak=f"{pk['rate_peak']:g}%", rate_peak_date=_date(pk["rate_peak_date"]), rate_low=f"{pk['rate_low']:g}%",
             rate_low_date=_date(pk["rate_low_date"]), npl_latest=f"{pk['npl_latest']:.1f}%", npl_latest_year=pk["npl_latest_year"], npl_peak=f"{pk['npl_peak']:.1f}%", npl_peak_year=pk["npl_peak_year"],
             npl_first=f"{pk['npl_first']:.1f}%", npl_first_year=pk["npl_first_year"], credit_latest=f"{pk['credit_latest']:.1f}%", credit_latest_year=pk["credit_latest_year"],
             n_policy_rows=len(r["policy"]))

    # --- tables and blocks
    v.update(metrics_table=metrics_table(r), comparisons_table=comparisons_table(r), scorecard_table=scorecard_table(r), scorecard_top_table=scorecard_table(r, 3), screening_table=screening_table(r),
             strategies_table=strategies_table(r), scenarios_table=scenarios_table(r), sensitivity_table=sensitivity_table(r), fairness_table=fairness_table(r),
             importance_table=importance_table(r), subsets_table=subsets_table(r), risk_bands_table=risk_bands_table(r), examples_table=examples_table(r),
             status_table=segment_table(r, "Repayment status last month (September)"), months_late_table=segment_table(r, "Months late in the last six"),
             limit_table=segment_table(r, "Credit limit"), data_dictionary_table=data_dictionary_table(), sources_table=sources_table(r), policy_table=policy_table(r),
             citation=config.UCI_CITATION, mcb_url=config.MCB_CARD_URL, uci_page=config.UCI_PAGE)
    return v


def build() -> None:
    v = values()
    targets = outputs()
    for template, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        text = render((TEMPLATE_DIR / template).read_text(encoding="utf-8"), v)
        target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    print(f"[docs] wrote {', '.join(str(t.relative_to(config.ROOT)) for t in targets.values())}")
