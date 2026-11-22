"""Turn the raw UCI file into one clean table of accounts, with the features the models use, and a record of what was found.

Nothing is dropped: the file has no missing values and its 35 repeated rows are mostly dormant cards, which are plausibly
different customers. What is done is written down in data/processed/data_quality.json:
  * EDUCATION codes 0, 5 and 6 and MARRIAGE code 0 are not in the documentation; they are put with "others";
  * accounts with identical features share a `group`, so that the train/test split never separates them;
  * the repayment status is read as "late" when it is 1 or more. Status 1 (one month late) hardly appears before September,
    so the earlier months are coded on a different basis from September and the numbers are not compared as exact months.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RECENT = 3                       # "recent" delinquency counts the latest three months
RATIO_CAP = 1.0                  # a payment above the bill counts as paying it in full
UTILISATION_CAP = 1.5            # a balance above 150% of the limit counts as 150%

ENGINEERED = [
    "utilisation_sep", "utilisation_avg", "utilisation_max", "bill_trend",
    "n_delinquent", "n_delinquent_recent", "months_since_delinquent", "worst_status", "n_revolving", "n_paid_full", "n_inactive",
    "status_trend", "pay_ratio_sep", "pay_ratio_aug", "pay_ratio_jul", "pay_ratio_jun", "pay_ratio_may",
    "pay_ratio_avg", "pay_ratio_total", "paid_total",
]
# Everything the account itself did or was given: its limit, six months of status, statements and payments, and what is made of them.
BEHAVIOUR = ["limit", *config.STATUS, *config.BILL, *config.PAID, *ENGINEERED]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """The raw file with readable column names. The columns must be exactly those the UCI file is documented to have."""
    raw = pd.read_csv(path or config.RAW / config.UCI_DATA_FILE)
    expected = ["ID", *config.UCI_NAMES, "Y"]
    if list(raw.columns) != expected:
        raise ValueError(f"unexpected columns in the raw file: {list(raw.columns)}")
    return raw.rename(columns={"ID": "id", **config.RENAME})


def validate(df: pd.DataFrame) -> None:
    """Raise if the table is not what the rest of the code assumes. The real file passes; a damaged one would not."""
    problems = []
    if df.isna().any().any():
        problems.append("missing values")
    if not df["id"].is_unique:
        problems.append("repeated ids")
    if not df["default"].isin([0, 1]).all():
        problems.append("default is not 0 or 1")
    if not df["sex_code"].isin(list(config.SEX_LABELS)).all():
        problems.append("sex code outside 1 and 2")
    if not df[config.STATUS].stack().between(-2, 9).all():
        problems.append("repayment status outside -2 to 9")
    if (df[config.PAID] < 0).any().any():
        problems.append("negative payments")
    if (df["limit"] <= 0).any():
        problems.append("a credit limit that is not positive")
    if not df["age"].between(18, 100).all():
        problems.append("age outside 18 to 100")
    if problems:
        raise ValueError("the data are not as expected: " + ", ".join(problems))


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Recode the codes the documentation does not explain, add readable labels and the duplicate groups.

    Returns the cleaned table and a dictionary of what was found, for data_quality.json.
    """
    validate(df)
    out = df.copy()
    quality: dict = {"rows": len(out), "default_rate": float(out["default"].mean()), "defaults": int(out["default"].sum())}

    quality["education_codes_before"] = {int(k): int(v) for k, v in out["education_code"].value_counts().sort_index().items()}
    quality["marriage_codes_before"] = {int(k): int(v) for k, v in out["marriage_code"].value_counts().sort_index().items()}
    quality["education_recoded"] = int(out["education_code"].isin(config.EDUCATION_RECODE).sum())
    quality["marriage_recoded"] = int(out["marriage_code"].isin(config.MARRIAGE_RECODE).sum())
    out["education_code"] = out["education_code"].replace(config.EDUCATION_RECODE)
    out["marriage_code"] = out["marriage_code"].replace(config.MARRIAGE_RECODE)
    unknown_education = set(out["education_code"]) - set(config.EDUCATION_LABELS)
    unknown_marriage = set(out["marriage_code"]) - set(config.MARRIAGE_LABELS)
    if unknown_education or unknown_marriage:
        raise ValueError(f"codes without a label: education {sorted(unknown_education)}, marriage {sorted(unknown_marriage)}")

    out["sex"] = out["sex_code"].map(config.SEX_LABELS)
    out["education"] = out["education_code"].map(config.EDUCATION_LABELS)
    out["marriage"] = out["marriage_code"].map(config.MARRIAGE_LABELS)
    out["age_band"] = pd.Series(pd.NA, index=out.index, dtype="object")
    for low, high, label in config.AGE_BANDS:
        out.loc[out["age"].between(low, high), "age_band"] = label
    if out["age_band"].isna().any():
        raise ValueError("an age falls outside every age band")

    features = [c for c in out.columns if c not in ("id", "default", "sex", "education", "marriage", "age_band")]
    out["group"] = out.groupby(features, sort=False).ngroup()
    sizes = out.groupby("group")["id"].transform("size")
    quality["rows_in_repeated_groups"] = int((sizes > 1).sum())
    quality["extra_copies"] = int(out.duplicated(subset=features).sum())
    dormant = (out[config.BILL] == 0).all(axis=1) & (out[config.PAID] == 0).all(axis=1)
    quality["repeated_rows_dormant"] = int(((sizes > 1) & dormant).sum())

    quality["accounts_with_a_negative_bill"] = int((out[config.BILL] < 0).any(axis=1).sum())
    quality["over_limit_in_september"] = int((out["bill_sep"] > out["limit"]).sum())
    quality["no_bills_in_six_months"] = int((out[config.BILL] == 0).all(axis=1).sum())
    quality["no_payments_in_six_months"] = int((out[config.PAID] == 0).all(axis=1).sum())
    quality["late_share_by_month"] = {m: float((out[f"status_{m}"] >= 1).mean()) for m in config.MONTHS}
    quality["status_one_by_month"] = {m: int((out[f"status_{m}"] == 1).sum()) for m in config.MONTHS}
    quality["status_two_by_month"] = {m: int((out[f"status_{m}"] == 2).sum()) for m in config.MONTHS}
    return out, quality


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """The behaviour measures the models use. Every one is worked out from the account's own six months of history."""
    out = df.copy()
    limit = out["limit"].astype(float)
    util = pd.DataFrame({m: (out[f"bill_{m}"] / limit).clip(0, UTILISATION_CAP) for m in config.MONTHS})
    out["utilisation_sep"] = util["sep"]
    out["utilisation_avg"] = util.mean(axis=1)
    out["utilisation_max"] = util.max(axis=1)
    out["bill_trend"] = util["sep"] - util["apr"]

    status = out[config.STATUS]
    late = status >= 1
    out["n_delinquent"] = late.sum(axis=1)
    out["n_delinquent_recent"] = late.iloc[:, :RECENT].sum(axis=1)
    first_late = late.to_numpy().argmax(axis=1)                        # position of the newest late month (0 = September)
    out["months_since_delinquent"] = np.where(late.any(axis=1), first_late, len(config.MONTHS))
    out["worst_status"] = status.max(axis=1)
    out["n_revolving"] = (status == 0).sum(axis=1)
    out["n_paid_full"] = (status == -1).sum(axis=1)
    out["n_inactive"] = (status == -2).sum(axis=1)
    out["status_trend"] = out["status_sep"] - out[["status_aug", "status_jul", "status_jun"]].mean(axis=1)

    # What share of the previous statement was paid? A payment in a month settles the statement of the month before, so the
    # latest ratio is September's payment over August's bill. With nothing owed, the ratio is 1.
    owed_total = paid_against = 0.0
    ratios = []
    for month, before in zip(config.MONTHS[:-1], config.MONTHS[1:], strict=True):
        owed = out[f"bill_{before}"]
        ratio = np.where(owed > 0, out[f"paid_{month}"] / owed.where(owed > 0, 1.0), 1.0)
        out[f"pay_ratio_{month}"] = np.clip(ratio, 0, RATIO_CAP)
        ratios.append(out[f"pay_ratio_{month}"])
        owed_total = owed_total + owed.clip(lower=0)
        paid_against = paid_against + np.minimum(out[f"paid_{month}"], owed.clip(lower=0))
    out["pay_ratio_avg"] = pd.concat(ratios, axis=1).mean(axis=1)
    out["pay_ratio_total"] = np.where(owed_total > 0, paid_against / np.where(owed_total > 0, owed_total, 1.0), 1.0)
    out["paid_total"] = out[config.PAID].sum(axis=1)
    return out


def run() -> None:
    raw = load_raw()
    accounts, quality = clean(raw)
    accounts = add_features(accounts)
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    accounts.to_csv(config.INTERIM / "accounts.csv.gz", index=False, compression="gzip", lineterminator="\n")
    (config.PROCESSED / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[build] {len(accounts):,} accounts, {int(accounts['default'].sum()):,} defaults ({accounts['default'].mean():.1%}); "
          f"{len(BEHAVIOUR)} behaviour columns")


def load_accounts() -> pd.DataFrame:
    """The table `run` wrote."""
    return pd.read_csv(config.INTERIM / "accounts.csv.gz")
