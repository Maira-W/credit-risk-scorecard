"""Constants shared by every stage: where the data come from, what the columns mean, and the rules of the analysis."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
DASHBOARD = ROOT / "dashboard"
NOTEBOOKS = ROOT / "notebooks"
TEMPLATES = Path(__file__).resolve().parent / "templates"
MANIFEST = RAW / "MANIFEST.json"

SEED = 42
USER_AGENT = "credit-risk-scorecard/1.0 (personal data-analysis portfolio project)"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# The cardholder data: UCI Machine Learning Repository, "Default of Credit Card Clients" (dataset 350), donated by I-Cheng Yeh.
# 30,000 credit card accounts of a bank in Taiwan, April to September 2005, licensed CC BY 4.0.
UCI_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
UCI_META_URL = "https://archive.ics.uci.edu/api/dataset?id=350"
UCI_DATA_FILE = "uci/default_of_credit_card_clients.csv"
UCI_META_FILE = "uci/metadata.json"
UCI_ROWS = 30_000
UCI_CITATION = ("Yeh, I-C. and Lien, C-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability "
                "of default of credit card clients. Expert Systems with Applications 36(2), 2473-2480. doi:10.1016/j.eswa.2007.12.020")

# Pakistan context: the banking sector's non-performing loans and the policy rate. World Bank data (CC BY 4.0). The World Bank's
# main series starts in 2013; its Global Financial Development database has the same ratio for 1998 to 2020.
WB_NPL_URL = "https://api.worldbank.org/v2/country/PAK/indicator/FB.AST.NPER.ZS?format=json&per_page=100"
WB_NPL_FILE = "macro/worldbank_npl_ratio.json"
WB_NPL_LONG_URL = "https://api.worldbank.org/v2/country/PAK/indicator/GFDD.SI.02?source=32&format=json&per_page=100"
WB_NPL_LONG_FILE = "macro/worldbank_gfdd_npl_ratio.json"
WB_CREDIT_URL = "https://api.worldbank.org/v2/country/PAK/indicator/FS.AST.PRVT.GD.ZS?format=json&per_page=100"
WB_CREDIT_FILE = "macro/worldbank_private_credit.json"
POLICY_RATE_FILE = REFERENCE / "policy_rate.csv"
MCB_CARD_URL = "https://www.mcb.com.pk/assets/documents/Credit-card-Summary-Box-1-July-to-Dec-16-Sep-2026-Font-11-English.pdf"

# ---------------------------------------------------------------------------
# The columns
# ---------------------------------------------------------------------------
# The file calls its columns X1 to X23 and Y. The UCI page's variable table gives each one the name used in the paper, and
# the paper's PAY_0 is September (there is no PAY_1). `test_data` checks this table against the saved UCI metadata.
UCI_NAMES = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3", "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3", "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}
MONTHS = ("sep", "aug", "jul", "jun", "may", "apr")            # newest first: the history ends in September 2005
MONTH_NAMES = {"sep": "September", "aug": "August", "jul": "July", "jun": "June", "may": "May", "apr": "April"}
# the name each column gets here: status_sep is the repayment status in September, bill_sep the September statement,
# paid_sep the payment made in September
RENAME = {"X1": "limit", "X2": "sex_code", "X3": "education_code", "X4": "marriage_code", "X5": "age", "Y": "default"}
for _i, _m in enumerate(MONTHS):
    RENAME[f"X{6 + _i}"] = f"status_{_m}"
    RENAME[f"X{12 + _i}"] = f"bill_{_m}"
    RENAME[f"X{18 + _i}"] = f"paid_{_m}"
STATUS = [f"status_{m}" for m in MONTHS]
BILL = [f"bill_{m}" for m in MONTHS]
PAID = [f"paid_{m}" for m in MONTHS]

SEX_LABELS = {1: "Male", 2: "Female"}
# EDUCATION is documented as 1 graduate school, 2 university, 3 high school, 4 others; the file also holds 0, 5 and 6, which the
# documentation does not explain (345 accounts). They are put with "others", which then holds 468 of the 30,000 accounts,
# rather than being guessed at.
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other or unknown"}
EDUCATION_RECODE = {0: 4, 5: 4, 6: 4}
# MARRIAGE is documented as 1 married, 2 single, 3 others; the file also holds 0 (54 accounts), put with "others".
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other or unknown"}
MARRIAGE_RECODE = {0: 3}
AGE_BANDS = ((21, 29, "21-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 120, "50 and over"))
DEMOGRAPHICS = ["sex_code", "education_code", "marriage_code", "age"]

STATUS_MEANING = {-2: "no spending", -1: "paid in full", 0: "paid the minimum (revolving)", 1: "1 month late", 2: "2 months late",
                  3: "3 months late", 4: "4 months late"}
_FEATURE_NAMES = {
    "limit": "Credit limit", "sex_code": "Sex", "education_code": "Education", "marriage_code": "Marital status", "age": "Age",
    "utilisation_sep": "Credit used, September", "utilisation_avg": "Credit used, six-month average", "utilisation_max": "Credit used, six-month peak",
    "bill_trend": "Change in credit used, April to September", "n_delinquent": "Months late, of six", "n_delinquent_recent": "Months late, of the last three",
    "months_since_delinquent": "Months since the last late payment", "worst_status": "Worst repayment status in six months",
    "n_revolving": "Months paying only the minimum", "n_paid_full": "Months paid in full", "n_inactive": "Months with no spending",
    "status_trend": "Repayment status, September against June to August", "pay_ratio_avg": "Share of the previous bill paid, average",
    "pay_ratio_total": "Share of the bills paid, five months", "paid_total": "Total paid, six months",
}


def feature_name(column: str) -> str:
    """A readable name for a column, for charts and documents."""
    if column in _FEATURE_NAMES:
        return _FEATURE_NAMES[column]
    kind, _, month = column.partition("_")
    if month in MONTH_NAMES:
        return {"status": "Repayment status, ", "bill": "Bill, ", "paid": "Payment, "}[kind] + MONTH_NAMES[month]
    if column.startswith("pay_ratio_") and column[len("pay_ratio_"):] in MONTH_NAMES:
        return "Share of the previous bill paid, " + MONTH_NAMES[column[len("pay_ratio_"):]]
    return column


# ---------------------------------------------------------------------------
# Splits, validation and the scorecard
# ---------------------------------------------------------------------------
HOLDOUT_SHARE = 0.20            # kept aside until every choice has been made
CV_FOLDS = 5
CV_REPEATS = 3
BOOTSTRAP_DRAWS = 1000
TUNING_TRIALS = 14               # random settings tried for the gradient-boosting challenger, each with 5-fold cross-validation
CALIBRATION_BINS = 10
DECILES = 10

# Binning (see creditrisk/binning.py): a bin must hold at least MIN_BIN_SHARE of the accounts, a feature has at most MAX_BINS
# bins, and neighbouring bins whose split adds less than IV_TOLERANCE of information value are merged.
MIN_BIN_SHARE = 0.05
MAX_BINS = 6
IV_TOLERANCE = 0.002
MIN_IV = 0.02                    # a feature below this has almost no power on its own and is dropped
MAX_CORRELATION = 0.60           # of two features whose WoE values correlate more than this, only the stronger is kept (see scorecard_variants.csv)
MAX_P_VALUE = 0.05               # a feature in the final regression must be significant at this level, with the expected sign

# Points: SCORE_BASE points at odds of SCORE_BASE_ODDS good accounts to 1 bad one, and SCORE_PDO points more every time the odds
# double. The default rate here is 22%, so odds of 20 to 1 (a 4.8% chance of default) is already a very good account.
SCORE_BASE = 600
SCORE_BASE_ODDS = 20
SCORE_PDO = 40

# The candidates for the scorecard: name, kind and the direction in which risk should move as the value rises ("up", "down",
# or None to let the data decide freely). A repayment status is an ordered code (-2 no spending, -1 paid in full, 0 paid the
# minimum, 1 and above months late), so it is binned in that order, with neighbouring codes merged but not forced to be monotone:
# paying in full is not safer than paying the minimum in this file.
SCORECARD_CANDIDATES = (
    ("status_sep", "numeric", None), ("status_aug", "numeric", None), ("status_jul", "numeric", None),
    ("status_jun", "numeric", None), ("status_may", "numeric", None), ("status_apr", "numeric", None),
    ("n_delinquent", "numeric", "up"), ("n_delinquent_recent", "numeric", "up"), ("months_since_delinquent", "numeric", "down"),
    ("worst_status", "numeric", "up"),
    ("utilisation_sep", "numeric", None), ("utilisation_avg", "numeric", None), ("utilisation_max", "numeric", None),
    ("pay_ratio_sep", "numeric", "down"), ("pay_ratio_avg", "numeric", "down"), ("pay_ratio_total", "numeric", "down"),
    ("limit", "numeric", "down"), ("bill_sep", "numeric", None), ("paid_sep", "numeric", "down"), ("paid_aug", "numeric", "down"),
    ("paid_total", "numeric", "down"),
)
# Five engineered measures are left out of the scorecard because a risk committee could not explain their bins: the change in
# utilisation (it has a spike at exactly zero), the change in status, and the counts of months paying the minimum, paying in full and
# not spending (their risk is not monotone, so a reason code such as "you paid the minimum 5 times, not 6" would make no sense).
# The challenger does use them. The month counts are tried once, in the scorecard variants table, to show what leaving them out costs.
MONTH_COUNT_CANDIDATES = (("n_revolving", "numeric", None), ("n_paid_full", "numeric", None), ("n_inactive", "numeric", None))
CORRELATION_CHECKS = (0.4, 0.5, 0.6, 0.7)
# What the gradient-boosting challenger may use (it takes every account-behaviour column, raw and engineered, but no demographics).
CHALLENGER_PARAMS = {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 100, "subsample": 0.8,
                     "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0}
CHALLENGER_THREADS = 2

# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
PROTECTED = ("sex", "age_band", "education", "marriage")
DISPARATE_IMPACT_MIN = 0.80      # the "four-fifths" screening rule: a group approved at under 80% of the best-treated group's rate is flagged
MIN_GROUP_SIZE = 100             # a group with fewer accounts in the hold-out is shown but not judged

# ---------------------------------------------------------------------------
# Economics: what an approved account earns and what a default costs, per unit of balance, over one billing month.
# Every number here is an assumption except where a source is named; the dashboard lets the reader change them all.
# ---------------------------------------------------------------------------
ECON_APR = 0.44                  # MCB's published retail markup: 3.67% a month, "an APR of up to 44%" (July to December 2026)
ECON_OPEX = 0.05                 # servicing, collection and fraud costs, a year, as a share of balances: an assumption
ECON_LOSS = 0.15                 # loss on an account that defaults, as a share of its balance: an assumption (see docs/methodology.md)
LOSS_CHECKS = (0.10, 0.15, 0.20, 0.30, 0.40)
APR_CHECKS = (0.30, 0.36, 0.44)
# The policy-rate dates shown as scenarios: the low before the 2021-23 tightening, the rate in mid-2022, the peak, the easing, now.
POLICY_SCENARIOS = ("2021-10-01", "2022-07-08", "2023-06-27", "2024-12-17", "2025-12-16", "2026-10-27")

# ---------------------------------------------------------------------------
# Palette shared by the figures and the dashboard
# ---------------------------------------------------------------------------
NAVY, TEAL, ORANGE, AMBER, GREY, PURPLE = "#1F3B57", "#2A9D8F", "#E76F51", "#E9C46A", "#9AA5B1", "#7B4FA3"
PLOTLY_JS = "https://cdn.plot.ly/plotly-3.5.0.min.js"
