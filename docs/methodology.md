# Method

How the scorecard was built, validated, audited and priced, in the order it was done. Every number is produced by the code in [`creditrisk/`](../creditrisk) and stored in [`data/processed`](../data/processed). To rebuild everything: `python -m creditrisk all`.

## 1. The question and the label

The question is a decision: **which cardholders should a lender keep extending credit to?** The label is the one in the data, "default payment next month": 1 if the account missed its payment in October 2005, 0 if it paid. 6,636 of 30,000 accounts (22.1%) are 1. The file does not define "missed" more closely than that, so it is read as *delinquent next month*, not as charged off. That matters for the economics (section 9).

The features are the six months before, April to September 2005: the credit limit, the repayment status each month, the bill and the payment each month, and four personal details (sex, age, education, marital status).

## 2. Cleaning

Nothing was dropped. [`creditrisk/build.py`](../creditrisk/build.py) checks the file is what it should be (no missing values, unique ids, an outcome of 0 or 1, valid codes) and then:

- **Recodes undocumented codes.** Education has codes 0, 5 and 6 that the documentation does not explain (345 accounts); marital status has 0 (54 accounts). They are put with "others", which then holds 468 and 377 accounts.
- **Groups identical accounts.** 108 accounts share features with another account (56 extra copies), 97 of them dormant cards with no bills or payments. They are kept, but the train/test split keeps each set of identical accounts on one side.
- **Reads repayment status as late or not.** The codes are -2 (no spending), -1 (paid in full), 0 (paid the minimum) and 1 or more (months late), but a status of 1 appears 3,688 times in September and only 28 times in August and 6 times in the four months before, while a status of 2 is common in every month (e.g. 3,927 in August). The older months seem to be coded on another basis, so the numbers are not compared as exact months, and no roll-rate table is built across the August to September boundary. The share of accounts that were late rises from 10.3% in April to 14.8% in August and 22.7% in September.
- **Notes, without changing, odd values.** 1,930 accounts have a negative bill in some month (a credit balance), 2,115 are over their limit in September, 866 had no bill at all in six months and 1,432 made no payment. Credit used is capped at 150% of the limit and floored at 0.

## 3. The split and the cross-validation

The accounts are split once, stratified on the outcome and grouped on identical accounts, into **24,000 development accounts (80%)** and **6,000 hold-out accounts (20%)**. Every choice is made on the development accounts: the bins, the features, the settings of the gradient-boosting model, the recalibration and the cut-off. The hold-out is used once, at the end, to report. Inside the development set, 3 repeats of 5-fold cross-validation compare the models; **every step is refitted inside each fold** (binning, feature choice, regression), so the cross-validated numbers include the cost of choosing the features from the data.

## 4. Features

20 measures are made from each account's own history (all in `build.add_features`, each checked by hand in the tests): credit used in September, on average and at its peak, as a share of the limit; the change in credit used; the number of months late (of six, and of the last three) and the months since the last late payment; the worst status; the number of months paying only the minimum, paying in full, and not spending; the share of the previous bill paid in each month, on average and over the five months together; and the total paid. The gradient-boosting model sees all of them and the raw columns; the scorecard sees a shortlist of them. Five measures whose bins a risk committee could not explain are left out of the scorecard's candidates: the change in credit used, the change in status, and the counts of months paying the minimum, paying in full and not spending (their risk is not monotone, so a reason code such as "you paid the minimum five times, not six" would make no sense). This was decided after a first run in which the count of months paying the minimum entered the card; including the three counts gives a cross-validated AUC of 0.772 against 0.769 without them (+0.0026), a gain too small to be worth a reason code nobody could explain.

## 5. The scorecard

**Weight of evidence.** A feature is cut into bins. Each bin gets WoE = ln(its share of all bad accounts ÷ its share of all good accounts): positive where the bin is riskier than average, negative where safer. The information value (IV) of a feature adds up (share of bad − share of good) × WoE over its bins; under 0.02 is useless, 0.1 to 0.3 medium, above 0.3 strong. Half an account is added to each bin so that a bin with no bad accounts still has a finite weight.

**Binning** ([`creditrisk/binning.py`](../creditrisk/binning.py)) starts with up to 30 small bins and repeatedly merges the neighbouring pair that loses least information, until every bin holds at least 5% of the accounts, there are at most 6 bins, a feature meant to be monotone is monotone, and no neighbouring pair could be merged for a loss of less than 0.002. Cut points are rounded to two significant figures of the feature's spread so that the card reads "45,000", not "44,873". Repayment statuses are binned in their natural order (-2, -1, 0, 1, ...) without forcing the risk to rise, because paying in full is not safer than paying the minimum in this file.

**Selection** ([`creditrisk/scorecard.py`](../creditrisk/scorecard.py)), on the development accounts only:

1. 21 candidate features are binned.
2. 1 are dropped for an information value under 0.02 or for having a single bin.
3. Of two features whose weights of evidence correlate above 0.60, only the one with the higher IV is kept (11 dropped). The limit was chosen by cross-validation on the development accounts: 0.4 gives 5.0 features and a cross-validated AUC of 0.761, 0.5 gives 7.2 and 0.767, 0.6 gives 7.2 and 0.769, and 0.7 gives 10.4 and 0.772, ([`scorecard_variants.csv`](../data/processed/scorecard_variants.csv)). The limit of 0.6 was chosen: against 0.7 it gives up 0.003 of cross-validated AUC, for a card with 3.2 fewer features on average.
4. A logistic regression is fitted to the weights of evidence, and features are dropped one at a time if their coefficient is not positive (a riskier-looking bin must never make an account look safer) or not significant at 5% (1 dropped).

8 features remain: Repayment status, September; Repayment status, July; Repayment status, May; Share of the previous bill paid, September; Total paid, six months; Payment, September; Credit limit; Credit used, six-month average. Everything dropped, with the reason, is in [`scorecard_screening.csv`](../data/processed/scorecard_screening.csv):

| Candidate | Information value | Decision | Why it was dropped |
| --- | ---: | --- | --- |
| Repayment status, September | 0.882 | kept |  |
| Months late, of six | 0.837 | dropped | correlated with status_sep (r = 0.75), which has the higher information value |
| Months late, of the last three | 0.831 | dropped | correlated with status_sep (r = 0.84), which has the higher information value |
| Months since the last late payment | 0.799 | dropped | correlated with status_sep (r = 0.85), which has the higher information value |
| Worst repayment status in six months | 0.738 | dropped | correlated with status_sep (r = 0.68), which has the higher information value |
| Repayment status, August | 0.560 | dropped | correlated with status_sep (r = 0.67), which has the higher information value |
| Repayment status, July | 0.410 | kept |  |
| Repayment status, June | 0.345 | dropped | correlated with status_jul (r = 0.62), which has the higher information value |
| Repayment status, May | 0.325 | kept |  |
| Repayment status, April | 0.282 | dropped | correlated with status_may (r = 0.65), which has the higher information value |
| Share of the previous bill paid, September | 0.229 | kept |  |
| Total paid, six months | 0.184 | kept |  |
| Payment, September | 0.177 | kept |  |
| Credit limit | 0.176 | kept |  |
| Payment, August | 0.152 | dropped | correlated with paid_total (r = 0.62), which has the higher information value |
| Credit used, six-month average | 0.138 | kept |  |
| Share of the bills paid, five months | 0.126 | dropped | coefficient -0.10 is not positive once the others are in the model |
| Share of the previous bill paid, average | 0.123 | dropped | correlated with pay_ratio_total (r = 0.89), which has the higher information value |
| Credit used, six-month peak | 0.099 | dropped | correlated with utilisation_avg (r = 0.81), which has the higher information value |
| Credit used, September | 0.083 | dropped | correlated with utilisation_avg (r = 0.74), which has the higher information value |
| Bill, September | 0.010 | dropped | information value under 0.02 |

**Points.** The score is `offset + factor × ln(odds of good against bad)`, with factor = 40 ÷ ln 2 so that every 40 points double the odds, and the offset chosen so that 600 points mean odds of 20 to 1 (a 4.8% probability of default). Each bin's points are `−factor × (coefficient × WoE + intercept ÷ number of features) + offset ÷ number of features`, rounded to whole points. The probability of default of an account is read from its whole-number score, so the card and the probability always agree; rounding changed the hold-out AUC by +0.0004. Scores on the hold-out run from 314 to 594.

**The card.**

| Feature | Bin | Accounts | Default rate | Points |
| --- | --- | ---: | ---: | ---: |
| **Repayment status, September** | -2 | 2,205 | 13.3% | 89 |
|  | -1 | 4,539 | 16.6% | 78 |
|  | 0 | 11,788 | 12.7% | 92 |
|  | 1 | 2,946 | 34.3% | 36 |
|  | 2 or more | 2,522 | 69.5% | -29 |
| **Total paid, six months** | 7,100 or less | 6,426 | 30.8% | 51 |
|  | 7,101 to 14,300 | 5,578 | 26.1% | 57 |
|  | 14,301 to 22,400 | 3,209 | 19.1% | 67 |
|  | 22,401 to 57,300 | 5,588 | 16.5% | 72 |
|  | 57,301 to 99,000 | 1,597 | 12.0% | 81 |
|  | 99,001 or more | 1,602 | 9.2% | 88 |
| **Repayment status, May** | -2 | 3,641 | 19.8% | 65 |
|  | -1 | 4,418 | 15.8% | 69 |
|  | 0 to 1 | 13,561 | 19.0% | 66 |
|  | 2 or more | 2,380 | 55.0% | 38 |
| **Repayment status, July** | -2 | 3,267 | 18.6% | 66 |
|  | -1 | 4,735 | 15.6% | 69 |
|  | 0 to 1 | 12,618 | 17.5% | 67 |
|  | 2 or more | 3,380 | 52.1% | 41 |
| **Credit used, six-month average** | up to 0.002 | 1,559 | 27.8% | 56 |
|  | over 0.002 up to 0.008 | 1,743 | 21.7% | 63 |
|  | over 0.008 up to 0.287 | 8,692 | 15.1% | 73 |
|  | over 0.287 up to 0.503 | 3,203 | 20.5% | 64 |
|  | over 0.503 up to 0.802 | 4,800 | 25.4% | 58 |
|  | over 0.802 | 4,003 | 32.6% | 51 |
| **Credit limit** | 30,000 or less | 3,237 | 35.9% | 52 |
|  | 30,001 to 70,000 | 4,150 | 27.7% | 58 |
|  | 70,001 to 140,000 | 4,862 | 23.6% | 61 |
|  | 140,001 to 300,000 | 8,088 | 16.9% | 67 |
|  | 300,001 or more | 3,663 | 13.1% | 72 |
| **Payment, September** | 0 | 4,205 | 36.2% | 55 |
|  | 1 to 3,900 | 11,817 | 22.1% | 62 |
|  | 3,901 to 8,910 | 4,777 | 16.2% | 66 |
|  | 8,911 or more | 3,201 | 12.5% | 69 |
| **Share of the previous bill paid, September** | up to 0 | 1,789 | 53.2% | 55 |
|  | over 0 up to 0.234 | 12,608 | 21.7% | 63 |
|  | over 0.234 | 9,603 | 16.8% | 64 |

## 6. The gradient-boosting challenger

LightGBM on the raw behaviour columns and the engineered measures (no personal details). 14 settings (leaves, minimum leaf size, learning rate, trees, column and row sampling, regularisation) are drawn at random, always including a sensible default, and the one with the lowest cross-validated log loss on the development accounts is kept ([`challenger_tuning.csv`](../data/processed/challenger_tuning.csv)). Its probabilities were already close to calibrated, so a logistic correction (slope 1.00, intercept 0.01), fitted on out-of-fold predictions of the development accounts, changes almost nothing: the hold-out log loss moves from 0.4231 to 0.4231. Probabilities are rounded to 0.1 of a percentage point so that the cut-off tables stay small. SHAP values (LightGBM's built-in tree contributions) show what it relies on:

| Feature | Share of the model's reliance | A higher value means |
| --- | ---: | --- |
| Repayment status, September | 10% | a code, not a quantity: late is riskier, but paying in full is not safer than paying the minimum |
| Months since the last late payment | 10% | less default |
| Worst repayment status in six months | 8% | more default |
| Months late, of the last three | 8% | more default |
| Bill, September | 6% | less default |
| Total paid, six months | 5% | less default |
| Months late, of six | 5% | more default |
| Credit limit | 4% | less default |
| Payment, August | 3% | less default |
| Credit used, September | 3% | more default |

## 7. Validation

All on the hold-out unless stated. **AUC**: the chance that a random defaulter scores riskier than a random payer. **Gini** = 2 × AUC − 1. **KS**: the largest gap between the cumulative score distributions of good and bad accounts. **Brier score**: mean squared error of the probabilities. **Calibration slope**: the slope of a logistic regression of the outcome on the log-odds of the prediction (1.00 is perfect; below 1 means too extreme). **Lift and capture**: default rate by tenth of the riskiest accounts and the share of all defaults found. **PSI**: population stability index of development against hold-out scores. **Bootstrap ranges**: 1,000 resamples of the hold-out accounts, with the same accounts drawn for every model so that differences are paired.

| Model | AUC, cross-validated | AUC, hold-out (95% range) | KS, hold-out (95% range) | Brier | Calibration slope | Defaults in riskiest tenth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive: last month's status only | 0.713 | **0.706** (0.688 to 0.723) | 0.356 (0.327 to 0.385) | 0.1427 | 0.98 | 32% |
| Scorecard | 0.769 | **0.768** (0.752 to 0.783) | 0.402 (0.379 to 0.432) | 0.1373 | 1.00 | 31% |
| Scorecard + sex, age, education, marriage | 0.769 | **0.768** (0.753 to 0.783) | 0.402 (0.379 to 0.432) | 0.1373 | 1.00 | 31% |
| Gradient boosting | 0.786 | **0.792** (0.777 to 0.806) | 0.443 (0.420 to 0.473) | 0.1332 | 1.01 | 32% |
| Gradient boosting + sex, age, education, marriage | 0.787 | **0.793** (0.778 to 0.807) | 0.444 (0.422 to 0.474) | 0.1328 | 1.02 | 32% |

| Comparison | AUC difference | 95% range | Verdict |
| --- | ---: | ---: | --- |
| Scorecard against naive: last month's status only | +0.0624 | +0.0490 to +0.0758 | a real difference |
| Gradient boosting against Scorecard | +0.0234 | +0.0170 to +0.0306 | a real difference |
| Scorecard + sex, age, education, marriage against Scorecard | +0.0004 | -0.0012 to +0.0020 | within the noise |
| Gradient boosting + sex, age, education, marriage against Gradient boosting | +0.0010 | -0.0009 to +0.0028 | within the noise |

Gradient boosting ranks better by 0.023 AUC (95% range 0.017 to 0.031), a difference that is larger than the noise of a 6,000-account test. Adding sex, age, education and marital status changes the scorecard's hold-out AUC by +0.0004 (95% range -0.0012 to +0.0020) and the gradient-boosting model's by +0.0010 (-0.0009 to +0.0028), so leaving them out costs almost nothing.

**The hard half.** A rule that reads only last month's status has nothing to go on for accounts that were not late, which is where 48% of the defaults are:

| Accounts | Model | Count | Default rate | AUC | Defaults in riskiest tenth |
| --- | --- | ---: | ---: | ---: | ---: |
| Not late in September | Naive: last month's status only | 4,650 | 14.2% | 0.532 | 13% |
| Not late in September | Scorecard | 4,650 | 14.2% | 0.664 | 20% |
| Not late in September | Gradient boosting | 4,650 | 14.2% | 0.706 | 25% |
| Late in September | Naive: last month's status only | 1,350 | 49.4% | 0.687 | 14% |
| Late in September | Scorecard | 1,350 | 49.4% | 0.738 | 16% |
| Late in September | Gradient boosting | 1,350 | 49.4% | 0.751 | 17% |

**Stability.** The population stability index of the scorecard score is 0.002, and the largest for any feature is 0.003. The split is random, so this only shows nothing is broken; there is no later period in the file, so stability over time cannot be tested.

| Band (scorecard probability) | Accounts | Default rate | Share of all defaults |
| --- | ---: | ---: | ---: |
| Very low risk (under 10%) | 23% | 7.5% | 8% |
| Low (10% to 15%) | 29% | 11.6% | 15% |
| Medium (15% to 25%) | 25% | 20.4% | 23% |
| High (25% to 50%) | 11% | 36.5% | 19% |
| Very high (50% and over) | 12% | 66.9% | 35% |

## 8. Fairness audit

Sex, age, education and marital status are not inputs to the scorecard or the challenger. They are used only to look at what the decisions do. At the cut-off, for each group: the **approval rate**; the **disparate-impact ratio** (approval rate over that of the best-treated group; under 0.80 is flagged, the "four-fifths" screening rule, which is not a Pakistani legal test); the share of **defaulters declined** and of **good accounts declined**; and **calibration** (average predicted probability over the observed default rate). Groups under 100 hold-out accounts are shown but not judged. At the break-even cut-off, 3 of 11 groups are approved at under four-fifths of the best-treated group's rate: age: 21-29, age: 50 and over and education: High school; for education: High school the whole 95% bootstrap range of the ratio is under 0.80, for the rest it reaches above it.

| Group (hold-out) | Accounts | Default rate | Predicted | Predicted ÷ observed | Approved | Default rate of approved | Disparate impact | 95% range of the ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sex: Female | 3,645 | 20.9% | 20.8% | 1.00 | 47% | 9.4% | 1.00 | 1.00 to 1.00 |
| Sex: Male | 2,355 | 24.1% | 23.6% | 0.98 | 41% | 8.8% | 0.86 | 0.82 to 0.92 |
| Age: 21-29 | 1,936 | 23.1% | 23.6% | 1.02 | 39% | 9.3% | 0.77 **flag** | 0.71 to 0.82 |
| Age: 30-39 | 2,222 | 20.1% | 20.2% | 1.01 | 51% | 8.3% | 1.00 | 1.00 to 1.00 |
| Age: 40-49 | 1,283 | 22.7% | 21.5% | 0.95 | 46% | 10.0% | 0.90 | 0.84 to 0.97 |
| Age: 50 and over | 559 | 25.4% | 23.5% | 0.93 | 38% | 10.7% | 0.76 **flag** | 0.67 to 0.84 |
| Education: Graduate school | 2,082 | 18.9% | 19.1% | 1.01 | 52% | 8.3% | 1.00 | 1.00 to 1.00 |
| Education: High school | 991 | 25.6% | 25.9% | 1.01 | 34% | 9.8% | 0.65 **flag** | 0.59 to 0.72 |
| Education: Other or unknown | 93 | 5.4% | 16.4% | 3.05 | 62% | 6.9% |  |  |
| Education: University | 2,834 | 23.8% | 22.7% | 0.96 | 42% | 9.9% | 0.81 | 0.76 to 0.85 |
| Marital status: Married | 2,718 | 23.4% | 21.9% | 0.93 | 45% | 10.2% | 1.00 | 0.95 to 1.00 |
| Marital status: Other or unknown | 69 | 23.2% | 24.4% | 1.05 | 33% | 13.0% |  |  |
| Marital status: Single | 3,213 | 21.0% | 21.9% | 1.04 | 45% | 8.2% | 1.00 | 0.94 to 1.00 |

Could the inputs act as a stand-in? A model that tries to tell the groups apart from the scorecard's inputs reaches an AUC of 0.60 for sex, 0.58 for being 40 or over and 0.61 for a university education (0.5 means the inputs say nothing about it). A model can be calibrated for every group and still approve groups at different rates when their true default rates differ; the audit shows both so that the two are not confused. This is a screening exercise, not legal advice.

## 9. The economics

Over one billing month, per unit of balance: an account that **pays** earns `margin = (yield − funding − running cost) ÷ 12`, and an account that **defaults** loses `loss`. The **yield** is the card APR times the share of balances that revolve; the APR is 44% (MCB's published retail markup, 3.67% a month, "an APR of up to 44%", July to December 2026, [source](https://www.mcb.com.pk/assets/documents/Credit-card-Summary-Box-1-July-to-Dec-16-Sep-2026-Font-11-English.pdf)), and 88% of balances are left unpaid after the next month's payment in this file (measured on the development accounts, over the five month pairs), so the yield is 38.8%. The **funding cost** is the policy rate (12% on 27 October 2026): money lent on a card is money not put in government paper. **Running cost** (5% a year) is an assumption. The **loss** (11.8% of the balance on an account that misses a payment) is measured from the roll rates of the repayment history (section 10), with two assumptions of its own. The margin is 1.82% a month (21.8% a year).

An account with probability of default p is worth `(1 − p) × margin − p × loss` per unit of balance, which is positive only while `p < margin ÷ (margin + loss)`: **13.4%**. A bad account costs 6.5 good accounts' monthly profit. With calibrated probabilities, approving every account at or below the break-even probability maximises expected profit, whatever the balances. The check on the data: the cut-off that earns most on the development accounts (out-of-fold predictions) is 14.8%, close to the break-even probability of 13.4%; it is the break-even probability that is reported, because it does not depend on a noisy peak.

Balances are the September statement, never below zero. Profit is on the hold-out accounts' real outcomes, per account in the whole portfolio (declined accounts earn nothing), with bootstrap ranges:

| Rule | Approved | Default rate of approved | Profit per account a month | 95% range | Better than approving everyone by |
| --- | ---: | ---: | ---: | ---: | ---: |
| Approve everyone | 100% | 22.1% | -NT$473 | -NT$607 to -NT$354 |  |
| Approve if not late last month (status 0 or better) | 78% | 14.2% | NT$89 | -NT$9 to NT$179 | NT$562 (NT$488 to NT$645) |
| Scorecard, at the break-even probability | 45% | 9.2% | NT$154 | NT$65 to NT$236 | NT$627 (NT$539 to NT$722) |
| Gradient boosting, at the break-even probability | 44% | 7.9% | NT$191 | NT$121 to NT$254 | NT$663 (NT$559 to NT$769) |

**Policy-rate scenarios.** The same calculation with the funding cost set to the policy rate on six dates, and with the first date's cut-off kept and tried under each later date's costs:

| Policy rate on | Rate | Margin a month | Break-even probability | Approved | Default rate of approved | Profit per account a month | If the Oct 2021 cut-off were kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 October 2021 | 7.25% | 2.21% | 15.9% | 56% | 10.2% | NT$263 | NT$263 |
| 8 July 2022 | 15% | 1.57% | 11.8% | 36% | 8.3% | NT$74 | NT$51 |
| 27 June 2023 | 22% | 0.99% | 7.7% | 7% | 4.0% | NT$23 | -NT$141 |
| 17 December 2024 | 13% | 1.74% | 12.9% | 42% | 9.1% | NT$122 | NT$106 |
| 16 December 2025 | 10.5% | 1.94% | 14.2% | 49% | 9.5% | NT$183 | NT$174 |
| 27 October 2026 | 12% | 1.82% | 13.4% | 45% | 9.2% | NT$154 | NT$133 |

**Sensitivity.** Share of accounts worth approving and the break-even probability, for combinations of APR and loss on a missed payment, at the current policy rate:

| Card APR, loss on a missed payment: | 4.8% | 6.8% | 11.8% | 15.0% | 30.0% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44% | 79% (under 27.6%) | 74% (under 21.0%) | 45% (under 13.4%) | 28% (under 10.8%) | 0% (under 5.7%) |
| 36% | 72% (under 20.5%) | 55% (under 15.3%) | 18% (under 9.5%) | 6% (under 7.6%) | 0% (under 3.9%) |
| 30% | 49% (under 14.2%) | 25% (under 10.4%) | 2% (under 6.3%) | 0% (under 5.0%) | 0% (under 2.6%) |

## 10. Roll rates and the loss on a missed payment

The label is a missed payment, so the loss on it cannot be read from the data. [`creditrisk/rollrates.py`](../creditrisk/rollrates.py) estimates it from how accounts move between repayment statuses.

- **States.** The five months April to August are the ones coded on one basis (a status of 1 hardly appears in them). A status of 0 or below is "current"; the late statuses 2, 3, 4, 5 and 6-or-more are five further states (a status of 1 joins 2).
- **Transition matrix.** Counting every account's move from one month to the next over the four month pairs gives 96,000 moves, and from them the chance of going from each state to each other state.

| This month, next month: | Current | Late 2 | Late 3 | Late 4 | Late 5 | Late 6+ | Moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 95% | 5% |  |  |  |  | 84,974 |
| Late 2 | 30% | 64% | 6% |  |  |  | 9,901 |
| Late 3 | 17% | 37% | 16% | 30% |  |  | 614 |
| Late 4 | 4% | 26% | 7% | 33% | 31% |  | 239 |
| Late 5 | 3% | 21% | 7% | 9% | 15% | 45% | 75 |
| Late 6+ | 1% | 25% | 2% | 1% | 1% | 71% | 197 |

- **From a missed payment to a write-off.** Treating the matrix as steady, raising it to the power of the number of months gives the chance that an account that starts in a given state reaches a given status within 12 months, counting it once it gets there.
- **Where the chain starts.** The label is a missed payment in October, so every account that defaulted is late in October, whatever it was in September (48% of them were current, 47% at the first late level, 5% deeper). Starting the chain from September's state would treat the current ones as not having missed yet, and let the late ones cure in the very step the label says they did not. Instead each is moved one month on, using the matrix's own odds of each late state from where it stood with the odds of being current set aside and the rest scaled up to one (`october_entry`): a current account lands on the first late level; a late one stays or gets worse. That puts 93% of the missed payments on the first late level, 5% one level deeper and 2% at or beyond the write-off line, and the chain runs from there. The 95% ranges resample accounts, so they cover the uncertainty of both the matrix and the starting point. The method is checked on simulated accounts that follow a known chain, where the true chance of a write-off after a missed payment can be counted (`tests/test_rollrates.py`).
- **A check against September.** The matrix was estimated on April to August, so the step from September to October is one it has not seen. Accounts at status 2 in September did not default 31% of the time, close to the 30% of late accounts that recovered in a month between April and August. Accounts at status 1 did not default 66% of the time: that status hardly appears before September, so the matrix has little to say about it. And accounts that were not late in September defaulted 13.8% of the time, against the 5.0% of current accounts that became late in an average month before. Either the crisis deepened in October or September's codes count lateness differently from the earlier months; the data cannot separate the two. Either way the roll rates of April to August probably understate how bad things got, which is one more reason to read the loss as a floor.
- **Loss.** loss = (share never recovered) x (chance of write-off) + (cost of collection and of the income lost while late). The share never recovered (85%) and the cost (2% of the balance) are assumptions; the chance is measured.

| Written off at | Chance within 12 months | 95% range | Loss | 95% range |
| --- | ---: | ---: | ---: | ---: |
| Status 4 or worse (base case) | 11.5% | 10.1% to 12.9% | 11.8% | 10.6% to 13.0% |
| Status 5 or worse | 5.7% | 4.5% to 6.8% | 6.8% | 5.8% to 7.8% |
| Status 6 or worse | 3.3% | 2.3% to 4.3% | 4.8% | 4.0% to 5.7% |

The loss depends on where the account stood. A first missed payment (an account that was current in September, 48% of the missed payments) is written off within 12 months 7.9% of the time, a loss of 8.7%; an account that was already late (52%), 14.8% of the time, a loss of 14.6%. The base case is the mix of the two, so a book of mostly current accounts would sit nearer the first figure.

| Where the account stood in September | Share of missed payments | Written off at status 4+ (loss) | Written off at status 5+ (loss) | Written off at status 6+ (loss) | 95% range, status 4+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current in September | 48% | 7.9% (8.7%) | 3.6% (5.1%) | 1.9% (3.6%) | 6.9% to 9.0% |
| Late in September | 52% | 14.8% (14.6%) | 7.6% (8.5%) | 4.5% (5.8%) | 13.1% to 16.6% |

The base case uses the conservative line (status 4) because the codes are not documented in months and the history is the start of a crisis, when roll rates were rising. 15%, the placeholder used before this was measured, is kept as a stress case. What each loss does to the cut-off, on the hold-out accounts:

| Loss case | Loss | Break-even probability | Scorecard approves | Scorecard profit per account a month (95% range) | Approving everyone | Gradient boosting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Roll rates, written off at status 6 | 4.8% | 27.6% | 79% | NT$441 (NT$388 to NT$489) | NT$248 | NT$434 |
| Roll rates, written off at status 5 | 6.8% | 21.0% | 74% | NT$337 (NT$273 to NT$398) | NT$36 | NT$338 |
| Roll rates, written off at status 4 (base case) | 11.8% | 13.4% | 45% | NT$154 (NT$65 to NT$236) | -NT$473 | NT$191 |
| Stress: 15% of the balance | 15.0% | 10.8% | 28% | NT$42 (-NT$57 to NT$138) | -NT$807 | NT$146 |
| Stress: 30% of the balance | 30.0% | 5.7% | 0% | -NT$2 (-NT$13 to NT$5) | -NT$2,354 | NT$71 |

At the lowest loss the roll rates support (4.8%), approving everyone earns NT$248 per account a month and the break-even probability is 27.6%, so 79% of accounts are worth approving; at the base case (11.8%) approving everyone loses NT$473 and 45% are worth approving; at the 15% stress loss only 28% are.

## 11. An explainable boosting model

LightGBM can be told that a feature may only push the probability of default one way. The constraints are in `config.MONOTONE`: credit limit, payments and the share of the bill paid can only lower the probability as they rise; months late, recent late months and the worst status can only raise it. Repayment statuses are codes and credit used is U-shaped at the low end, so they are left free. The same settings as the free model are used and the probabilities are recalibrated the same way. Whether the model obeys is checked, not assumed: for a sample of accounts each constrained feature is moved across its range with everything else fixed, and any move of the probability the wrong way is counted.

| Model | AUC (95% range) | KS | Brier | Calibration slope | Profit per account at the break-even cut-off (95% range) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scorecard | 0.768 (0.752 to 0.783) | 0.402 | 0.1373 | 1.00 | NT$154 (NT$65 to NT$236) |
| Monotone boosting | 0.792 (0.777 to 0.806) | 0.455 | 0.1331 | 1.02 | NT$185 (NT$118 to NT$246) |
| Gradient boosting | 0.792 (0.777 to 0.806) | 0.443 | 0.1332 | 1.01 | NT$191 (NT$121 to NT$254) |

A boosting model told which way 19 features must push the probability reaches an AUC of 0.792: level with free boosting's (95% range of the gap -0.0015 to +0.0016; within the noise) and 0.023 above the scorecard's (+0.0173 to +0.0304; a real gap). At the break-even cut-off it earns NT$185 per account a month: NT$31 more than the scorecard's NT$154 (95% range of the gap -NT$27 to NT$93; a difference that could be chance on a test this size), against NT$191 for free boosting. Moving each constrained feature across its range for 300 sampled accounts (5,700 account-and-feature checks), free boosting moved the wrong way somewhere along the range in 4,191 of them (74%) and the monotone model in none; where free boosting did, the probability fell below an earlier level by 0.6 percentage points on average and by as much as 9.2 percentage points.

## 12. Swap sets

A swap-set analysis, standard in model validation, compares two models by the accounts on which they disagree. At the break-even cut-off, each account is approved by both models, by one, or by neither; for each group the table shows how many, how many defaulted, their average balance, and what approving them would have earned (realised, per account of the whole portfolio).

**Scorecard against gradient boosting**

| Who approves | Accounts | Defaulted | Average balance | Profit if approved |
| --- | ---: | ---: | ---: | ---: |
| Both approve | 37.6% | 7.6% | NT$65,534 | NT$183 |
| Only scorecard approves | 7.1% | 17.3% | NT$113,995 | -NT$29 |
| Only gradient boosting approves | 6.6% | 9.6% | NT$34,703 | NT$8 |
| Neither approves | 48.7% | 35.7% | NT$32,639 | -NT$634 |

**Scorecard against monotone boosting**

| Who approves | Accounts | Defaulted | Average balance | Profit if approved |
| --- | ---: | ---: | ---: | ---: |
| Both approve | 37.0% | 7.6% | NT$64,404 | NT$177 |
| Only scorecard approves | 7.6% | 16.8% | NT$116,198 | -NT$23 |
| Only monotone boosting approves | 5.3% | 9.8% | NT$36,334 | NT$8 |
| Neither approves | 50.0% | 35.0% | NT$32,522 | -NT$634 |

**Monotone boosting against gradient boosting**

| Who approves | Accounts | Defaulted | Average balance | Profit if approved |
| --- | ---: | ---: | ---: | ---: |
| Both approve | 41.4% | 7.8% | NT$60,947 | NT$195 |
| Only monotone boosting approves | 0.9% | 11.1% | NT$59,225 | -NT$10 |
| Only gradient boosting approves | 2.8% | 9.7% | NT$60,628 | -NT$4 |
| Neither approves | 54.9% | 33.7% | NT$42,768 | -NT$653 |

The scorecard and gradient boosting decide alike on 86% of accounts. Of the 13.7% they split on, those only gradient boosting approves defaulted at 9.6% and carry an average balance of NT$34,703, against 17.3% and NT$113,995 for those only the scorecard approves; approving the first group would have earned NT$8 per account of the portfolio, the second -NT$29.

## 13. Stress tests

A stress test asks what a shock does to profit. Defaults are raised by multiplying every account's odds of default by a factor (for small probabilities this is close to multiplying the probability), money gets dearer by taking the funding cost from the policy rate at its 2023 peak, or both. For each scenario the expected profit per account is worked out from the scorecard's own probabilities for the accounts approved under two policies: the cut-off kept where it is, and the cut-off re-set to the scenario's break-even. These are expected values, not outcomes: the file has no accounts that lived through a different year.

| Scenario | Policy rate | Average default probability | Cut-off kept: approved | Kept: expected profit | Re-set: approved | Re-set: expected profit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base case | 12% | 21.9% | 45% | NT$161 | 45% | NT$161 |
| Defaults 25% higher | 12% | 25.1% | 45% | NT$66 | 30% | NT$81 |
| Defaults 50% higher | 12% | 27.9% | 45% | -NT$25 | 17% | NT$35 |
| Defaults double | 12% | 32.7% | 45% | -NT$194 | 5% | NT$5 |
| Policy rate at its 2023 peak | 22% | 21.9% | 45% | -NT$85 | 7% | NT$5 |
| 2023 peak rate and defaults 50% higher | 22% | 27.9% | 45% | -NT$260 | 0% | NT$0 |

With defaults 50% higher and the cut-off kept, expected profit goes from NT$161 to -NT$25 a month; re-setting the cut-off gives NT$35 on 17% of accounts. At the 2023 peak policy rate the kept cut-off gives -NT$85 and a re-set one NT$5; with defaults 50% higher as well, -NT$260 against NT$0.

## 14. Risk-based pricing

The same arithmetic can be turned round. The margin on a good account must cover the expected loss, p x loss / (1 - p), so the APR that makes approving a group break even is (12 x that margin + funding + running cost) / (share of balances that revolve). For each tenth of accounts, from the safest:

| Tenth (1 = safest) | Average default probability | Average balance | Break-even APR | Pays at the current APR | Expected profit per account a month |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 7.1% | NT$75,014 | 32% | yes | NT$624 |
| 2 | 9.0% | NT$80,734 | 35% | yes | NT$482 |
| 3 | 10.4% | NT$78,836 | 38% | yes | NT$329 |
| 4 | 11.7% | NT$67,675 | 41% | yes | NT$159 |
| 5 | 13.5% | NT$43,356 | 44% | no | NT$1 |
| 6 | 15.5% | NT$35,856 | 49% | no | -NT$97 |
| 7 | 17.8% | NT$18,423 | 54% | no | -NT$109 |
| 8 | 23.1% | NT$20,864 | 67% | no | -NT$265 |
| 9 | 40.6% | NT$36,734 | 129% | no | -NT$1,537 |
| 10 | 70.1% | NT$51,873 | 394% | no | -NT$3,873 |

At the current card APR of 44%, approving pays for the safest 4 of 10 tenths of accounts. The other 6 would need APRs from 44% to 394% to break even. The riskiest need prices far above the one card rate checked here (MCB's published 44%), so for them the choice is to decline or to limit the exposure, not to price for risk.

## 15. Moving the scorecard to another lender

A scorecard fitted on accounts that miss a payment at one rate can be used on a lender whose accounts miss at another. The ranking is kept; every log-odds moves by logit(new rate) - logit(old rate), the usual correction for a change in the base rate (it assumes the way the features relate to risk is the same, which is the part that needs a local test). The break-even cut-off, in the original probabilities, is then the break-even probability moved back by the same amount.

| Monthly missed-payment rate of the lender's accounts | Cut-off score | Share approved | Expected default rate of approved | Expected profit per account a month |
| ---: | ---: | ---: | ---: | ---: |
| 5.0% | 438 | 87% | 3.4% | NT$638 |
| 8.0% | 467 | 82% | 4.8% | NT$546 |
| 12.0% | 493 | 77% | 6.7% | NT$431 |
| 22.1% (this file) | 535 | 45% | 9.9% | NT$161 |

A lender whose accounts miss a payment 8.0% of the time, instead of 22.1%, would put the cut-off at 467 points instead of 535 and approve 82% of accounts instead of 45%, with the same economics and the same ranking.

## 16. What drives the approval gaps

The scorecard is a sum of points, so the gap in average score between a group and the best-treated group of its attribute is the sum of the gaps in the points each feature gives. The table covers every group that is flagged or within 10 points of the four-fifths line (disparate impact under 0.90) and lists the three features that contribute most to its gap.

| Group | Approved | Reference approved | Disparate impact | Average points gap | Biggest contributors (points) |
| --- | ---: | ---: | ---: | ---: | --- |
| Sex: Male against Female | 41% | 47% | 0.86 | -8.8 | Repayment status, September (-4.0); Credit used, six-month average (-1.2); Credit limit (-0.9) |
| Age: 21-29 against 30-39 | 39% | 51% | 0.77 | -12.6 | Credit limit (-3.6); Repayment status, September (-3.0); Total paid, six months (-2.7) |
| Age: 50 and over against 30-39 | 38% | 51% | 0.76 | -12.0 | Repayment status, September (-4.8); Total paid, six months (-2.2); Credit used, six-month average (-2.1) |
| Education: High school against Graduate school | 34% | 52% | 0.65 | -22.9 | Repayment status, September (-6.8); Credit limit (-4.2); Total paid, six months (-3.3) |
| Education: University against Graduate school | 42% | 52% | 0.81 | -12.7 | Credit limit (-3.1); Repayment status, September (-2.4); Credit used, six-month average (-2.3) |

The biggest single contributor to each group's gap is sex Male: Repayment status, September (-4.0 of -8.8 points); age 21-29: Credit limit (-3.6 of -12.6 points); age 50 and over: Repayment status, September (-4.8 of -12.0 points); education High school: Repayment status, September (-6.8 of -22.9 points); education University: Credit limit (-3.1 of -12.7 points). No single feature accounts for more than 46% of any of these gaps: they build up across the card.

## 17. Limits of the method

- The cardholders are not Pakistani and the book is from a crisis period; nothing here measures how Pakistani cardholders behave.
- "Default" is a missed payment. The loss on it comes from roll rates over five months at the start of a crisis, with two assumptions (the share never recovered, the cost of collection); the running cost and (to some degree) the APR that applies are assumptions too. The re-anchoring to another lender assumes the features relate to risk in the same way there.
- One cohort: no out-of-time test, no view of behaviour through a cycle.
- Selection: only accounts the bank had approved are in the data.
- One month of margin is counted, which understates the value of an account that keeps paying.
- The fairness audit looks at four groups on the information in the file; it does not cover other groups or other notions of fairness, and it is not legal advice.
- The statistical ranges describe sampling noise in a 6,000-account test, not uncertainty about the assumptions.

## 18. Reproducibility

A fixed seed (42) drives the split, the cross-validation, the tuning and the bootstrap. Library versions are pinned in `requirements.txt`. Raw files are checked against `data/raw/MANIFEST.json` by `python -m creditrisk verify`. The tests check each function on made-up portfolios with known answers (weight of evidence, points, the break-even cut-off), run the whole analysis on a sample, and check that the committed results, documents, dashboard and notebook agree with each other.
