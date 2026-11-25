# Model card: card default scorecard

A short, standard description of the model: what it is for, how well it works, for whom it works less well, and what to watch. It follows the usual model-card headings. The model is a portfolio exercise on public data, not a deployed system.

## Model details

- **What it is.** A points scorecard: 8 features in 37 bins (from 21 candidates), weights of evidence fed to a logistic regression, turned into whole points. A score of 600 means odds of 20 good accounts to 1 bad one, and every 40 points double the odds. Higher is safer. Scores on the hold-out run from 314 to 594.
- **The features.** Repayment status, September; Repayment status, July; Repayment status, May; Share of the previous bill paid, September; Total paid, six months; Payment, September; Credit limit; Credit used, six-month average.
- **A challenger.** A LightGBM gradient-boosting model on all the behaviour columns, recalibrated; used for comparison, not for decisions.
- **Version and date.** 1.0.0, November 2026. Code: this repository (`creditrisk/`), MIT licence.
- **Contact.** The repository's owner.

## Intended use

- **Intended.** Ranking existing cardholders by their chance of missing next month's payment, and approving credit (a new facility or a limit increase) for those above a cut-off score, as a demonstration of how a bank would do it. Each decline comes with the three features that cost the account most points (`Scorecard.reason_codes`).
- **Out of scope.** Application scoring of people with no card history, any use on Pakistani borrowers without refitting, lending to a person on this model alone, any decision about a person's sex, age, religion or other personal status, and any use as evidence of how Pakistani cardholders behave.

## Data

30,000 accounts of a bank in Taiwan, April to September 2005; the outcome is whether the account missed its payment in October 2005. 24,000 accounts (80%) were used to build and tune everything, 6,000 (20%) were kept for one final test. Cleaning and quirks: [data sources](data_sources.md). Default rate: 22.1%.

## Performance (hold-out)

| Model | AUC, cross-validated | AUC, hold-out (95% range) | KS, hold-out (95% range) | Brier | Calibration slope | Defaults in riskiest tenth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive: last month's status only | 0.713 | **0.706** (0.688 to 0.723) | 0.356 (0.327 to 0.385) | 0.1427 | 0.98 | 32% |
| Scorecard | 0.769 | **0.768** (0.752 to 0.783) | 0.402 (0.379 to 0.432) | 0.1373 | 1.00 | 31% |
| Scorecard + sex, age, education, marriage | 0.769 | **0.768** (0.753 to 0.783) | 0.402 (0.379 to 0.432) | 0.1373 | 1.00 | 31% |
| Gradient boosting | 0.786 | **0.792** (0.777 to 0.806) | 0.443 (0.420 to 0.473) | 0.1332 | 1.01 | 32% |
| Gradient boosting + sex, age, education, marriage | 0.787 | **0.793** (0.778 to 0.807) | 0.444 (0.422 to 0.474) | 0.1328 | 1.02 | 32% |

- AUC 0.768, Gini 0.536, KS 0.402. The cross-validated AUC on the development accounts, with binning and feature choice refitted in every fold, is 0.769 (standard deviation 0.008), so the hold-out is typical.
- Gradient boosting ranks better by 0.023 AUC (95% range 0.017 to 0.031), a difference that is larger than the noise of a 6,000-account test.
- Calibration: slope 1.00 (1.00 is perfect). The largest gap between the average predicted and the observed default rate in any tenth of accounts is 3.5 percentage points (`data/processed/calibration.csv`).
- **Accounts that were not late.** Among the 4,650 hold-out accounts that were not late last month, the AUC is 0.66, against 0.77 overall; accounts that have not yet shown trouble leave less to see.

| Accounts | Model | Count | Default rate | AUC | Defaults in riskiest tenth |
| --- | --- | ---: | ---: | ---: | ---: |
| Not late in September | Naive: last month's status only | 4,650 | 14.2% | 0.532 | 13% |
| Not late in September | Scorecard | 4,650 | 14.2% | 0.664 | 20% |
| Not late in September | Gradient boosting | 4,650 | 14.2% | 0.706 | 25% |
| Late in September | Naive: last month's status only | 1,350 | 49.4% | 0.687 | 14% |
| Late in September | Scorecard | 1,350 | 49.4% | 0.738 | 16% |
| Late in September | Gradient boosting | 1,350 | 49.4% | 0.751 | 17% |

| Band (scorecard probability) | Accounts | Default rate | Share of all defaults |
| --- | ---: | ---: | ---: |
| Very low risk (under 10%) | 23% | 7.5% | 8% |
| Low (10% to 15%) | 29% | 11.6% | 15% |
| Medium (15% to 25%) | 25% | 20.4% | 23% |
| High (25% to 50%) | 11% | 36.5% | 19% |
| Very high (50% and over) | 12% | 66.9% | 35% |

**An explainable alternative.** A boosting model told which way 19 features must push the probability reaches an AUC of 0.792: level with free boosting's (95% range of the gap -0.0015 to +0.0016; within the noise) and 0.023 above the scorecard's (+0.0173 to +0.0304; a real gap). At the break-even cut-off it earns NT$185 per account a month: NT$31 more than the scorecard's NT$154 (95% range of the gap -NT$27 to NT$93; a difference that could be chance on a test this size), against NT$191 for free boosting. Moving each constrained feature across its range for 300 sampled accounts (5,700 account-and-feature checks), free boosting moved the wrong way somewhere along the range in 4,191 of them (74%) and the monotone model in none; where free boosting did, the probability fell below an earlier level by 0.6 percentage points on average and by as much as 9.2 percentage points.

**Where the scorecard and gradient boosting disagree.** The scorecard and gradient boosting decide alike on 86% of accounts. Of the 13.7% they split on, those only gradient boosting approves defaulted at 9.6% and carry an average balance of NT$34,703, against 17.3% and NT$113,995 for those only the scorecard approves; approving the first group would have earned NT$8 per account of the portfolio, the second -NT$29.

Three hold-out accounts and the reasons the card gave:

| Hold-out account | Score | Probability of default | Defaulted? | Reasons (points lost against the best bin) |
| --- | ---: | ---: | --- | --- |
| Riskier than 90% of accounts | 415 | 55.2% | yes | Repayment status, September: 2 or more (-121); Total paid, six months: 14,301 to 22,400 (-21); Credit used, six-month average: over 0.503 up to 0.802 (-15) |
| In the middle | 529 | 14.6% | no | Total paid, six months: 7,100 or less (-37); Credit limit: 70,001 to 140,000 (-11); Credit used, six-month average: over 0.287 up to 0.503 (-9) |
| Safer than 90% of accounts | 567 | 8.1% | no | Total paid, six months: 14,301 to 22,400 (-21); Payment, September: 1 to 3,900 (-7); Share of the previous bill paid, September: over 0 up to 0.234 (-1) |

## Fairness

Sex, age, education and marital status are **not** inputs. They are audited at the cut-off (535, a probability of 13.4%):

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

At the break-even cut-off, 3 of 11 groups are approved at under four-fifths of the best-treated group's rate: age: 21-29, age: 50 and over and education: High school; for education: High school the whole 95% bootstrap range of the ratio is under 0.80, for the rest it reaches above it. The flagged groups have higher default rates in this file than the best-treated group of the same attribute, and the model is calibrated for them (predicted ÷ observed between 0.93 and 1.02), so the gaps in approval mostly follow gaps in observed default rates, not a model that is harsher than the data on those groups. That does not make the gaps acceptable: under the four-fifths screening rule they would need a closer look, for example whether a different feature set or cut-off narrows them at a small cost. Women default less than men in this file (20.8% against 24.2%) and are approved more often (47% against 41%); the model is calibrated for both (predicted ÷ observed 1.00 for women, 0.98 for men).

Could the inputs act as a stand-in for the protected attributes? A model that tries to tell the groups apart from the scorecard's inputs reaches an AUC of 0.60 for sex, 0.58 for being 40 or over and 0.61 for a university education (0.5 is no information): the inputs carry some signal about each, as behaviour usually does. Adding sex, age, education and marital status changes the scorecard's hold-out AUC by +0.0004 (95% range -0.0012 to +0.0020) and the gradient-boosting model's by +0.0010 (-0.0009 to +0.0028), so leaving them out costs almost nothing.

What lies behind the gaps: the scorecard is a sum of points, so a group's gap splits by feature (looked at here for the groups that are flagged or within 10 points of the line, disparate impact under 0.90). The biggest single contributor to each group's gap is sex Male: Repayment status, September (-4.0 of -8.8 points); age 21-29: Credit limit (-3.6 of -12.6 points); age 50 and over: Repayment status, September (-4.8 of -12.0 points); education High school: Repayment status, September (-6.8 of -22.9 points); education University: Credit limit (-3.1 of -12.7 points). No single feature accounts for more than 46% of any of these gaps: they build up across the card.

This is a screening exercise on one data set, not legal advice, and the four-fifths rule is a US convention, not a Pakistani legal test.

## Limits and risks

- **Not Pakistani data**, and from a crisis-era book (22.1% missed a payment in one month).
- **"Default" is a missed payment.** The loss on it is estimated from roll rates over five months at the start of a crisis (a floor, not a forecast), with two assumptions; the data cannot say more.
- **One cohort, no out-of-time test.** The population stability index between development and hold-out is 0.002, which only shows the split is random.
- **Selection.** Only accounts the bank had approved are in the data; applicants it rejected are not.
- **Repayment codes** are only partly documented (see data sources).
- **The cut-off depends on assumptions** about yield, cost and loss that are not in the data. Across the sensitivity grid the break-even probability runs from 2.6% to 27.6%.

## Monitoring if it were deployed

| What | How often | Warning | Action |
|---|---|---|---|
| Population stability index of the score | monthly | above 0.10 | investigate; above 0.25 refit |
| Default rate of each risk band against its prediction | monthly | outside the band's 95% range | recalibrate |
| AUC and KS on matured accounts | quarterly | fall of more than 0.03 | refit |
| Approval rate and default rate by group | whenever the cut-off changes | any group under four-fifths of the best group's approval rate | review features and cut-off |
| Cut-off against the policy rate | whenever the State Bank changes its rate | the break-even probability moves by 1 percentage point or more | reset the cut-off |
| Roll rates behind the loss | quarterly | the chance that a missed payment reaches status 4 within a year moves by 2 percentage points or more from 11.5% | re-estimate the loss, then reset the cut-off |
| Directions of the explainable model | at every refit | any account whose probability moves the wrong way in the check of `advanced.monotone_violations` | do not deploy until it passes |
