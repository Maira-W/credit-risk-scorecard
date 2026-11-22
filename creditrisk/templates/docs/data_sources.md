# Data sources

Every file the analysis reads, where it came from, and what was done to it. All raw files are open data (CC BY 4.0) and are stored in [`data/raw`](../data/raw), so the project runs without a download. [`data/raw/MANIFEST.json`](../data/raw/MANIFEST.json) records the source, time, size and SHA-256 of each file, and `python -m creditrisk verify` checks them.

{{sources_table}}

## The cardholder data

**What it is.** "Default of Credit Card Clients", UCI Machine Learning Repository dataset 350 ([page]({{uci_page}})), donated by I-Cheng Yeh. {{accounts:,}} credit card accounts of a bank in Taiwan: each account's credit limit, sex, education, marital status and age, its repayment status, bill and payment in each of six months (April to September 2005), and whether it defaulted on its payment the next month (October 2005).

**Licence and credit.** Creative Commons Attribution 4.0 (CC BY 4.0), which allows sharing and adaptation with credit. Cite as: {{citation}}

**Why this dataset.** I looked for public borrower-level credit data for Pakistan and found none, so the borrowers are a foreign stand-in. It is a well-known, documented, citable file with a real default outcome and enough history per account to build behavioural features. Its age (2005) and place (Taiwan, at the start of a card-debt crisis) are its main limits: see the README.

**The columns.** The file calls them X1 to X23 and Y. The names in the first column below come from UCI's variable table, and `tests/test_data.py` checks them against the saved metadata. Month 1 in the paper's numbering is September (the paper's `PAY_0`).

{{data_dictionary_table}}

**What was found and done** (all in `creditrisk/build.py`; the record is `data/processed/data_quality.json`):

- No missing values; ids 1 to 30,000 are unique; the outcome is 0 or 1. The file is checked for this every time it is read.
- Education codes 0, 5 and 6 and marital status code 0 are not in the documentation ({{education_recoded}} and {{marriage_recoded}} accounts). They are put with "others".
- {{repeated_rows}} accounts have the same features as another account ({{extra_copies}} extra copies); {{repeated_dormant}} of them are dormant cards with no bills or payments. They are kept and never split between training and test.
- Repayment status: a status of 1 ("one month late") appears {{status_one_sep:,}} times in September and {{status_one_aug}} times in August, but only {{status_one_rest}} times in the four months before, while 2 is common in every month. The older months look to be coded on another basis, so the code is read as "late" (1 or more) and not as an exact number of months. The share of accounts that were late goes from {{late_apr_pct}} in April to {{late_aug_pct}} in August and {{late_sep_pct}} in September.
- {{negative_bill_accounts:,}} accounts have a negative bill in some month (a credit balance), {{over_limit:,}} are over their limit in September, {{no_bills}} had no bill in any of the six months and {{no_payments:,}} made no payment in any of them. Nothing is dropped; credit used is floored at 0 and capped at 150% of the limit.
- The documentation does not say exactly what "default payment next month" means. It is read as missing the payment due next month, not as a charge-off.

## Pakistan: the policy rate

[`data/reference/policy_rate.csv`](../data/reference/policy_rate.csv): the State Bank of Pakistan's policy rate, {{n_policy_rows}} changes from {{rate_low_date}} ({{rate_low}}) to {{rate_now_date}} ({{rate_now}}), peaking at {{rate_peak}} on {{rate_peak_date}}. It is typed in by hand from news reports of the State Bank's announcements, with a source for every row (the State Bank's own site does not allow automated reading, as found when building an earlier project). The date is the effective date where one was reported, otherwise the day after the announcement; each row says which.

{{policy_table}}

## Pakistan: banks' bad loans and credit

World Bank series, fetched through the World Bank's API (CC BY 4.0), used as context only (they do not enter any model):

- **Bank non-performing loans to gross loans**, %, Pakistan: the main series (FB.AST.NPER.ZS, {{npl_latest_year}}: {{npl_latest}}) starts in 2013; the Global Financial Development database (GFDD.SI.02) has the same ratio for 1998 to 2020 ({{npl_first_year}}: {{npl_first}}; peak {{npl_peak}} in {{npl_peak_year}}). The two are identical in the years they share (checked in `tests/test_data.py`), and the project joins them.
- **Domestic credit to the private sector**, % of GDP (FS.AST.PRVT.GD.ZS): {{credit_latest}} in {{credit_latest_year}}.

These are not comparable with the {{default_rate_pct}} in the cardholder data: that is accounts that missed one payment, this is loans that have become non-performing, a much later and much rarer event.

## Pakistan: the card price

The yield on the card is set from MCB's published credit card summary box, effective 1 July to 31 December 2026 ([source]({{mcb_url}})): a service fee/markup of 3.67% per month on retail purchases and cash advances, "translated into an APR of up to 44%", charged when the balance is not paid in full by the due date. That is one bank's published rate and an upper figure, so the economics treat it as an assumption with a range (APR 30% to 44%), not as a fact about every card.

## What I could not get

- Public borrower-level data from Pakistan, or any Pakistani card portfolio's default or loss rates.
- Loss given default and recoveries for cards, anywhere: the loss on a default in the economics is an assumption, shown with a range.
- The State Bank of Pakistan's own tables by script: its site returns an error to automated requests, so its figures come from news reports of its announcements.
- Any data after October 2005 for the cardholders, so there is no out-of-time test.
