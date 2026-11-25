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

Three hold-out accounts and the reasons the card gave:

| Hold-out account | Score | Probability of default | Defaulted? | Reasons (points lost against the best bin) |
| --- | ---: | ---: | --- | --- |
| Riskier than 90% of accounts | 415 | 55.2% | yes | Repayment status, September: 2 or more (-121); Total paid, six months: 14,301 to 22,400 (-21); Credit used, six-month average: over 0.503 up to 0.802 (-15) |
| In the middle | 529 | 14.6% | no | Total paid, six months: 7,100 or less (-37); Credit limit: 70,001 to 140,000 (-11); Credit used, six-month average: over 0.287 up to 0.503 (-9) |
| Safer than 90% of accounts | 567 | 8.1% | no | Total paid, six months: 14,301 to 22,400 (-21); Payment, September: 1 to 3,900 (-7); Share of the previous bill paid, September: over 0 up to 0.234 (-1) |

## Fairness

Sex, age, education and marital status are **not** inputs. They are audited at the cut-off (549, a probability of 10.8%):

| Group (hold-out) | Accounts | Default rate | Predicted | Predicted ÷ observed | Approved | Default rate of approved | Disparate impact | 95% range of the ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sex: Female | 3,645 | 20.9% | 20.8% | 1.00 | 30% | 8.2% | 1.00 | 1.00 to 1.00 |
| Sex: Male | 2,355 | 24.1% | 23.6% | 0.98 | 26% | 6.4% | 0.85 | 0.78 to 0.92 |
| Age: 21-29 | 1,936 | 23.1% | 23.6% | 1.02 | 23% | 5.7% | 0.66 **flag** | 0.59 to 0.73 |
| Age: 30-39 | 2,222 | 20.1% | 20.2% | 1.01 | 34% | 7.6% | 1.00 | 1.00 to 1.00 |
| Age: 40-49 | 1,283 | 22.7% | 21.5% | 0.95 | 29% | 8.5% | 0.86 | 0.77 to 0.95 |
| Age: 50 and over | 559 | 25.4% | 23.5% | 0.93 | 23% | 10.9% | 0.67 **flag** | 0.57 to 0.79 |
| Education: Graduate school | 2,082 | 18.9% | 19.1% | 1.01 | 37% | 6.7% | 1.00 | 1.00 to 1.00 |
| Education: High school | 991 | 25.6% | 25.9% | 1.01 | 18% | 7.9% | 0.48 **flag** | 0.41 to 0.55 |
| Education: Other or unknown | 93 | 5.4% | 16.4% | 3.05 | 42% | 7.7% |  |  |
| Education: University | 2,834 | 23.8% | 22.7% | 0.96 | 25% | 8.4% | 0.68 **flag** | 0.62 to 0.74 |
| Marital status: Married | 2,718 | 23.4% | 21.9% | 0.93 | 29% | 9.3% | 1.00 | 0.94 to 1.00 |
| Marital status: Other or unknown | 69 | 23.2% | 24.4% | 1.05 | 10% | 14.3% |  |  |
| Marital status: Single | 3,213 | 21.0% | 21.9% | 1.04 | 28% | 6.0% | 0.98 | 0.91 to 1.00 |

At the break-even cut-off, 4 of 11 groups are approved at under four-fifths of the best-treated group's rate: age: 21-29, age: 50 and over, education: High school and education: University; for every one of them the whole 95% bootstrap range of the ratio is under 0.80, so the gaps are unlikely to be chance. The flagged groups have higher default rates in this file than the best-treated group of the same attribute, and the model is calibrated for them (predicted ÷ observed between 0.93 and 1.02), so the gaps in approval mostly follow gaps in observed default rates, not a model that is harsher than the data on those groups. That does not make the gaps acceptable: under the four-fifths screening rule they would need a closer look, for example whether a different feature set or cut-off narrows them at a small cost. Women default less than men in this file (20.8% against 24.2%) and are approved more often (30% against 26%); the model is calibrated for both (predicted ÷ observed 1.00 for women, 0.98 for men).

Could the inputs act as a stand-in for the protected attributes? A model that tries to tell the groups apart from the scorecard's inputs reaches an AUC of 0.60 for sex, 0.58 for being 40 or over and 0.61 for a university education (0.5 is no information): the inputs carry some signal about each, as behaviour usually does. Adding sex, age, education and marital status changes the scorecard's hold-out AUC by +0.0004 (95% range -0.0012 to +0.0020) and the gradient-boosting model's by +0.0010 (-0.0009 to +0.0028), so leaving them out costs almost nothing.

This is a screening exercise on one data set, not legal advice, and the four-fifths rule is a US convention, not a Pakistani legal test.

## Limits and risks

- **Not Pakistani data**, and from a crisis-era book (22.1% missed a payment in one month).
- **"Default" is a missed payment.** The economics assume what a default costs; the data cannot say.
- **One cohort, no out-of-time test.** The population stability index between development and hold-out is 0.002, which only shows the split is random.
- **Selection.** Only accounts the bank had approved are in the data; applicants it rejected are not.
- **Repayment codes** are only partly documented (see data sources).
- **The cut-off depends on assumptions** about yield, cost and loss that are not in the data. Across the sensitivity grid the break-even probability runs from 1.9% to 15.4%.

## Monitoring if it were deployed

| What | How often | Warning | Action |
|---|---|---|---|
| Population stability index of the score | monthly | above 0.10 | investigate; above 0.25 refit |
| Default rate of each risk band against its prediction | monthly | outside the band's 95% range | recalibrate |
| AUC and KS on matured accounts | quarterly | fall of more than 0.03 | refit |
| Approval rate and default rate by group | whenever the cut-off changes | any group under four-fifths of the best group's approval rate | review features and cut-off |
| Cut-off against the policy rate | whenever the State Bank changes its rate | the break-even probability moves by 1 percentage point or more | reset the cut-off |
