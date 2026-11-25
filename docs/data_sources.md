# Data sources

Every file the analysis reads, where it came from, and what was done to it. All raw files are open data (CC BY 4.0) and are stored in [`data/raw`](../data/raw), so the project runs without a download. [`data/raw/MANIFEST.json`](../data/raw/MANIFEST.json) records the source, time, size and SHA-256 of each file, and `python -m creditrisk verify` checks them.

| File in data/raw | What it is | Licence | Bytes | Retrieved | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| `uci/default_of_credit_card_clients.csv` | 30,000 credit card accounts in Taiwan, April to September 2005 (UCI dataset 350) | CC BY 4.0 | 2,897,080 | 2026-11-22 | `45bcf4df62ff...` |
| `uci/metadata.json` | UCI's description of the dataset: variables, notes, citation | CC BY 4.0 | 7,253 | 2026-11-22 | `b92ff7a69d5b...` |
| `macro/worldbank_npl_ratio.json` | Pakistan: bank non-performing loans to gross loans, % (World Bank FB.AST.NPER.ZS) | CC BY 4.0 | 15,290 | 2026-11-22 | `60326fb4e2fa...` |
| `macro/worldbank_gfdd_npl_ratio.json` | Pakistan: bank non-performing loans to gross loans, %, 1998 to 2020 (World Bank GFDD.SI.02) | CC BY 4.0 | 13,747 | 2026-11-22 | `9d195ff41a75...` |
| `macro/worldbank_private_credit.json` | Pakistan: domestic credit to the private sector, % of GDP (World Bank FS.AST.PRVT.GD.ZS) | CC BY 4.0 | 15,790 | 2026-11-22 | `b2e644deccfb...` |

## The cardholder data

**What it is.** "Default of Credit Card Clients", UCI Machine Learning Repository dataset 350 ([page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)), donated by I-Cheng Yeh. 30,000 credit card accounts of a bank in Taiwan: each account's credit limit, sex, education, marital status and age, its repayment status, bill and payment in each of six months (April to September 2005), and whether it defaulted on its payment the next month (October 2005).

**Licence and credit.** Creative Commons Attribution 4.0 (CC BY 4.0), which allows sharing and adaptation with credit. Cite as: Yeh, I-C. and Lien, C-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications 36(2), 2473-2480. doi:10.1016/j.eswa.2007.12.020

**Why this dataset.** I looked for public borrower-level credit data for Pakistan and found none, so the borrowers are a foreign stand-in. It is a well-known, documented, citable file with a real default outcome and enough history per account to build behavioural features. Its age (2005) and place (Taiwan, at the start of a card-debt crisis) are its main limits: see the README.

**The period.** April to October 2005 is the start of Taiwan's card-debt crisis: banks had issued cards widely, and an editorial in the Taipei Times of 27 December 2005 already describes tens of thousands of people deep in credit-card debt ("Like 'go west,' credit debt's a threat", [link](https://www.taipeitimes.com/News/editorials/archives/2005/12/27/2003286251)). That helps explain why 22.1% of these accounts missed a payment in a single month: treat that as a crisis-era rate, not a typical one.

**The columns.** The file calls them X1 to X23 and Y. The names in the first column below come from UCI's variable table, and `tests/test_data.py` checks them against the saved metadata. Month 1 in the paper's numbering is September (the paper's `PAY_0`).

| File column | UCI name | Name here | Meaning |
| --- | --- | --- | --- |
| ID | ID | id | Account number, 1 to 30,000 |
| X1 | LIMIT_BAL | limit | Credit limit in NT$ (individual and family credit) |
| X2 | SEX | sex_code | 1 male, 2 female |
| X3 | EDUCATION | education_code | 1 graduate school, 2 university, 3 high school, 4 other |
| X4 | MARRIAGE | marriage_code | 1 married, 2 single, 3 other |
| X5 | AGE | age | Age in years |
| X6 | PAY_0 | status_sep | Repayment status in September 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late |
| X7 | PAY_2 | status_aug | Repayment status in August 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late |
| X8 | PAY_3 | status_jul | Repayment status in July 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late |
| X9 | PAY_4 | status_jun | Repayment status in June 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late |
| X10 | PAY_5 | status_may | Repayment status in May 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late |
| X11 | PAY_6 | status_apr | Repayment status in April 2005: -2 no spending, -1 paid in full, 0 paid the minimum, 1 and above months late |
| X12 | BILL_AMT1 | bill_sep | Bill statement in September 2005, NT$ |
| X13 | BILL_AMT2 | bill_aug | Bill statement in August 2005, NT$ |
| X14 | BILL_AMT3 | bill_jul | Bill statement in July 2005, NT$ |
| X15 | BILL_AMT4 | bill_jun | Bill statement in June 2005, NT$ |
| X16 | BILL_AMT5 | bill_may | Bill statement in May 2005, NT$ |
| X17 | BILL_AMT6 | bill_apr | Bill statement in April 2005, NT$ |
| X18 | PAY_AMT1 | paid_sep | Payment made in September 2005, NT$ |
| X19 | PAY_AMT2 | paid_aug | Payment made in August 2005, NT$ |
| X20 | PAY_AMT3 | paid_jul | Payment made in July 2005, NT$ |
| X21 | PAY_AMT4 | paid_jun | Payment made in June 2005, NT$ |
| X22 | PAY_AMT5 | paid_may | Payment made in May 2005, NT$ |
| X23 | PAY_AMT6 | paid_apr | Payment made in April 2005, NT$ |
| Y | default payment next month | default | 1 if the account missed its payment in October 2005 |

**What was found and done** (all in `creditrisk/build.py`; the record is `data/processed/data_quality.json`):

- No missing values; ids 1 to 30,000 are unique; the outcome is 0 or 1. The file is checked for this every time it is read.
- Education codes 0, 5 and 6 and marital status code 0 are not in the documentation (345 and 54 accounts). They are put with "others".
- 108 accounts have the same features as another account (56 extra copies); 97 of them are dormant cards with no bills or payments. They are kept and never split between training and test.
- Repayment status: a status of 1 ("one month late") appears 3,688 times in September and 28 times in August, but only 6 times in the four months before, while 2 is common in every month. The older months look to be coded on another basis, so the code is read as "late" (1 or more) and not as an exact number of months. The share of accounts that were late goes from 10.3% in April to 14.8% in August and 22.7% in September.
- 1,930 accounts have a negative bill in some month (a credit balance), 2,115 are over their limit in September, 866 had no bill in any of the six months and 1,432 made no payment in any of them. Nothing is dropped; credit used is floored at 0 and capped at 150% of the limit.
- The documentation does not say exactly what "default payment next month" means. It is read as missing the payment due next month, not as a charge-off.
- The roll rates use only the five months April to August, which are coded on one basis; September, where a status of 1 suddenly appears, is used only to say where the accounts that missed their payment stood.

## Pakistan: the policy rate

[`data/reference/policy_rate.csv`](../data/reference/policy_rate.csv): the State Bank of Pakistan's policy rate, 21 changes from 1 October 2021 (7.25%) to 27 October 2026 (12%), peaking at 22% on 27 June 2023. It is typed in by hand from news reports of the State Bank's announcements, with a source for every row (the State Bank's own site does not allow automated reading, as found when building an earlier project). The date is the effective date where one was reported, otherwise the day after the announcement; each row says which.

| Effective | Policy rate |
| --- | ---: |
| 1 October 2021 | 7.25% |
| 22 November 2021 | 8.75% |
| 15 December 2021 | 9.75% |
| 8 April 2022 | 12.25% |
| 24 May 2022 | 13.75% |
| 8 July 2022 | 15% |
| 26 November 2022 | 16% |
| 24 January 2023 | 17% |
| 3 March 2023 | 20% |
| 5 April 2023 | 21% |
| 27 June 2023 | 22% |
| 11 June 2024 | 20.5% |
| 30 July 2024 | 19.5% |
| 13 September 2024 | 17.5% |
| 5 November 2024 | 15% |
| 17 December 2024 | 13% |
| 28 January 2025 | 12% |
| 6 May 2025 | 11% |
| 16 December 2025 | 10.5% |
| 28 April 2026 | 11.5% |
| 27 October 2026 | 12% |

## Pakistan: banks' bad loans and credit

World Bank series, fetched through the World Bank's API (CC BY 4.0), used as context only (they do not enter any model):

- **Bank non-performing loans to gross loans**, %, Pakistan: the main series (FB.AST.NPER.ZS, 2025: 5.4%) starts in 2013; the Global Financial Development database (GFDD.SI.02) has the same ratio for 1998 to 2020 (1998: 19.5%; peak 23.4% in 2001). The two are identical in the years they share (checked in `tests/test_data.py`), and the project joins them.
- **Domestic credit to the private sector**, % of GDP (FS.AST.PRVT.GD.ZS): 10.7% in 2025.

These are not comparable with the 22.1% in the cardholder data: that is accounts that missed one payment, this is loans that have become non-performing, a much later and much rarer event.

## Pakistan: the card price

The yield on the card is set from MCB's published credit card summary box, effective 1 July to 31 December 2026 ([source](https://www.mcb.com.pk/assets/documents/Credit-card-Summary-Box-1-July-to-Dec-16-Sep-2026-Font-11-English.pdf)): a service fee/markup of 3.67% per month on retail purchases and cash advances, "translated into an APR of up to 44%", charged when the balance is not paid in full by the due date. That is one bank's published rate and an upper figure, so the economics treat it as an assumption with a range (APR 30% to 44%), not as a fact about every card.

## What I could not get

- Public borrower-level data from Pakistan, or any Pakistani card portfolio's default or loss rates.
- Loss given default and recoveries for cards, anywhere: the loss in the economics is built from the roll rates of the file's own repayment history, with two assumptions (the share of a written-off balance never recovered, the cost of collection), and shown with a range.
- The State Bank of Pakistan's own tables by script: its site returns an error to automated requests, so its figures come from news reports of its announcements.
- Any data after October 2005 for the cardholders, so there is no out-of-time test.
