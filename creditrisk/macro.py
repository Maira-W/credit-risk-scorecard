"""The Pakistan side: the policy rate, and the banking sector's non-performing loans and private credit.

These are context, not inputs to the models. The cardholder data are from Taiwan; what ties the project to Pakistan is the cost of
money (the policy rate sets the break-even cut-off) and how risky lending has been in the banks the cut-off would be used by.
"""

from __future__ import annotations

import pandas as pd

from . import config, ingest


def policy_rate() -> pd.DataFrame:
    return pd.read_csv(config.POLICY_RATE_FILE, parse_dates=["effective_date"]).sort_values("effective_date").reset_index(drop=True)


def _npl_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    recent, _ = ingest.parse_worldbank((config.RAW / config.WB_NPL_FILE).read_text(encoding="utf-8"))
    long, _ = ingest.parse_worldbank((config.RAW / config.WB_NPL_LONG_FILE).read_text(encoding="utf-8"))
    return recent, long


def npl_ratio() -> pd.DataFrame:
    """Bank non-performing loans to gross loans, % a year: the World Bank's Global Financial Development figures for the years up
    to 2012 and its main series from 2013 on. In the years both have, they are the same (checked in the tests)."""
    recent, long = _npl_parts()
    recent["source"] = "World Bank (main series)"
    long["source"] = "World Bank (Global Financial Development)"
    earlier = long[~long["year"].isin(recent["year"])]
    return pd.concat([earlier, recent], ignore_index=True).sort_values("year").reset_index(drop=True)


def npl_overlap() -> pd.DataFrame:
    """The years in which both series have a value, with the two values side by side."""
    recent, long = _npl_parts()
    return recent.merge(long, on="year", suffixes=("_main", "_long"))


def private_credit() -> pd.DataFrame:
    series, _ = ingest.parse_worldbank((config.RAW / config.WB_CREDIT_FILE).read_text(encoding="utf-8"))
    return series


def context() -> dict:
    """The facts about Pakistan that the documents quote."""
    rate, npl, credit = policy_rate(), npl_ratio(), private_credit()
    peak = rate.loc[rate["policy_rate_pct"].idxmax()]
    low = rate[rate["effective_date"] >= "2021-01-01"].iloc[0]
    latest_npl, peak_npl = npl.iloc[-1], npl.loc[npl["value"].idxmax()]
    return {
        "rate_now": float(rate.iloc[-1]["policy_rate_pct"]), "rate_now_date": str(rate.iloc[-1]["effective_date"].date()),
        "rate_peak": float(peak["policy_rate_pct"]), "rate_peak_date": str(peak["effective_date"].date()),
        "rate_low": float(low["policy_rate_pct"]), "rate_low_date": str(low["effective_date"].date()),
        "npl_latest": float(latest_npl["value"]), "npl_latest_year": int(latest_npl["year"]),
        "npl_peak": float(peak_npl["value"]), "npl_peak_year": int(peak_npl["year"]),
        "npl_first_year": int(npl.iloc[0]["year"]), "npl_first": float(npl.iloc[0]["value"]),
        "credit_latest": float(credit.iloc[-1]["value"]), "credit_latest_year": int(credit.iloc[-1]["year"]),
        "credit_first_year": int(credit.iloc[0]["year"]), "credit_first": float(credit.iloc[0]["value"]),
    }
