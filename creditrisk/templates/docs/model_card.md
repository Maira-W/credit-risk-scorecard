# Model card: card default scorecard

A short, standard description of the model: what it is for, how well it works, for whom it works less well, and what to watch. It follows the usual model-card headings. The model is a portfolio exercise on public data, not a deployed system.

## Model details

- **What it is.** A points scorecard: {{n_features}} features in {{n_bins}} bins (from {{n_candidates}} candidates), weights of evidence fed to a logistic regression, turned into whole points. A score of {{base_score}} means odds of {{base_odds}} good accounts to 1 bad one, and every {{pdo}} points double the odds. Higher is safer. Scores on the hold-out run from {{score_min}} to {{score_max}}.
- **The features.** {{features_list}}.
- **A challenger.** A LightGBM gradient-boosting model on all the behaviour columns, recalibrated; used for comparison, not for decisions.
- **Version and date.** 1.0.0, November 2026. Code: this repository (`creditrisk/`), MIT licence.
- **Contact.** The repository's owner.

## Intended use

- **Intended.** Ranking existing cardholders by their chance of missing next month's payment, and approving credit (a new facility or a limit increase) for those above a cut-off score, as a demonstration of how a bank would do it. Each decline comes with the three features that cost the account most points (`Scorecard.reason_codes`).
- **Out of scope.** Application scoring of people with no card history, any use on Pakistani borrowers without refitting, lending to a person on this model alone, any decision about a person's sex, age, religion or other personal status, and any use as evidence of how Pakistani cardholders behave.

## Data

{{accounts:,}} accounts of a bank in Taiwan, April to September 2005; the outcome is whether the account missed its payment in October 2005. {{n_dev:,}} accounts (80%) were used to build and tune everything, {{n_hold:,}} (20%) were kept for one final test. Cleaning and quirks: [data sources](data_sources.md). Default rate: {{default_rate_pct}}.

## Performance (hold-out)

{{metrics_table}}

- AUC {{auc_scorecard:.3f}}, Gini {{gini_scorecard:.3f}}, KS {{ks_scorecard:.3f}}. The cross-validated AUC on the development accounts, with binning and feature choice refitted in every fold, is {{cv_auc_scorecard:.3f}} (standard deviation {{cv_auc_scorecard_sd:.3f}}), {{cv_hold_phrase}}.
- {{gap_sentence}}.
- Calibration: slope {{slope_scorecard:.2f}} (1.00 is perfect). The largest gap between the average predicted and the observed default rate in any tenth of accounts is {{cal_gap_pp}} percentage points (`data/processed/calibration.csv`).
- **Accounts that were not late.** Among the {{hard_accounts:,}} hold-out accounts that were not late last month, the AUC is {{hard_auc_scorecard:.2f}}, against {{auc_scorecard:.2f}} overall{{hard_phrase}}.

{{subsets_table}}

{{risk_bands_table}}

Three hold-out accounts and the reasons the card gave:

{{examples_table}}

## Fairness

Sex, age, education and marital status are **not** inputs. They are audited at the cut-off ({{cutoff_score}}, a probability of {{break_even}}):

{{fairness_table}}

{{fairness_headline}} {{fairness_reading}} That does not make the gaps acceptable: under the four-fifths screening rule they would need a closer look, for example whether a different feature set or cut-off narrows them at a small cost. {{sex_sentence}}

Could the inputs act as a stand-in for the protected attributes? A model that tries to tell the groups apart from the scorecard's inputs reaches an AUC of {{proxy_sex:.2f}} for sex, {{proxy_age:.2f}} for being 40 or over and {{proxy_edu:.2f}} for a university education (0.5 is no information): the inputs carry some signal about each, as behaviour usually does. {{demographics_sentence}}.

This is a screening exercise on one data set, not legal advice, and the four-fifths rule is a US convention, not a Pakistani legal test.

## Limits and risks

- **Not Pakistani data**, and from a crisis-era book ({{default_rate_pct}} missed a payment in one month).
- **"Default" is a missed payment.** The economics assume what a default costs; the data cannot say.
- **One cohort, no out-of-time test.** The population stability index between development and hold-out is {{psi_score:.3f}}, which only shows the split is random.
- **Selection.** Only accounts the bank had approved are in the data; applicants it rejected are not.
- **Repayment codes** are only partly documented (see data sources).
- **The cut-off depends on assumptions** about yield, cost and loss that are not in the data. Across the sensitivity grid the break-even probability runs from {{sens_be_min}} to {{sens_be_max}}.

## Monitoring if it were deployed

| What | How often | Warning | Action |
|---|---|---|---|
| Population stability index of the score | monthly | above 0.10 | investigate; above 0.25 refit |
| Default rate of each risk band against its prediction | monthly | outside the band's 95% range | recalibrate |
| AUC and KS on matured accounts | quarterly | fall of more than 0.03 | refit |
| Approval rate and default rate by group | whenever the cut-off changes | any group under four-fifths of the best group's approval rate | review features and cut-off |
| Cut-off against the policy rate | whenever the State Bank changes its rate | the break-even probability moves by 1 percentage point or more | reset the cut-off |
