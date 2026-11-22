# Method

How the scorecard was built, validated, audited and priced, in the order it was done. Every number is produced by the code in [`creditrisk/`](../creditrisk) and stored in [`data/processed`](../data/processed). To rebuild everything: `python -m creditrisk all`.

## 1. The question and the label

The question is a decision: **which cardholders should a lender keep extending credit to?** The label is the one in the data, "default payment next month": 1 if the account missed its payment in October 2005, 0 if it paid. {{defaults:,}} of {{accounts:,}} accounts ({{default_rate_pct}}) are 1. The file does not define "missed" more closely than that, so it is read as *delinquent next month*, not as charged off. That matters for the economics (section 9).

The features are the six months before, April to September 2005: the credit limit, the repayment status each month, the bill and the payment each month, and four personal details (sex, age, education, marital status).

## 2. Cleaning

Nothing was dropped. [`creditrisk/build.py`](../creditrisk/build.py) checks the file is what it should be (no missing values, unique ids, an outcome of 0 or 1, valid codes) and then:

- **Recodes undocumented codes.** Education has codes 0, 5 and 6 that the documentation does not explain ({{education_recoded}} accounts); marital status has 0 ({{marriage_recoded}} accounts). They are put with "others", which then holds {{education_other_total}} and {{marriage_other_total}} accounts.
- **Groups identical accounts.** {{repeated_rows}} accounts share features with another account ({{extra_copies}} extra copies), {{repeated_dormant}} of them dormant cards with no bills or payments. They are kept, but the train/test split keeps each set of identical accounts on one side.
- **Reads repayment status as late or not.** The codes are -2 (no spending), -1 (paid in full), 0 (paid the minimum) and 1 or more (months late), but a status of 1 appears {{status_one_sep:,}} times in September and only {{status_one_aug}} times in August and {{status_one_rest}} times in the four months before, while a status of 2 is common in every month (e.g. {{status_two_aug:,}} in August). The older months seem to be coded on another basis, so the numbers are not compared as exact months, and no roll-rate table is built across the August to September boundary. The share of accounts that were late rises from {{late_apr_pct}} in April to {{late_aug_pct}} in August and {{late_sep_pct}} in September.
- **Notes, without changing, odd values.** {{negative_bill_accounts:,}} accounts have a negative bill in some month (a credit balance), {{over_limit:,}} are over their limit in September, {{no_bills}} had no bill at all in six months and {{no_payments:,}} made no payment. Credit used is capped at 150% of the limit and floored at 0.

## 3. The split and the cross-validation

The accounts are split once, stratified on the outcome and grouped on identical accounts, into **{{n_dev:,}} development accounts (80%)** and **{{n_hold:,}} hold-out accounts (20%)**. Every choice is made on the development accounts: the bins, the features, the settings of the gradient-boosting model, the recalibration and the cut-off. The hold-out is used once, at the end, to report. Inside the development set, {{cv_repeats}} repeats of {{cv_folds}}-fold cross-validation compare the models; **every step is refitted inside each fold** (binning, feature choice, regression), so the cross-validated numbers include the cost of choosing the features from the data.

## 4. Features

{{n_engineered}} measures are made from each account's own history (all in `build.add_features`, each checked by hand in the tests): credit used in September, on average and at its peak, as a share of the limit; the change in credit used; the number of months late (of six, and of the last three) and the months since the last late payment; the worst status; the number of months paying only the minimum, paying in full, and not spending; the share of the previous bill paid in each month, on average and over the five months together; and the total paid. The gradient-boosting model sees all of them and the raw columns; the scorecard sees a shortlist of them. Five measures whose bins a risk committee could not explain are left out of the scorecard's candidates: the change in credit used, the change in status, and the counts of months paying the minimum, paying in full and not spending (their risk is not monotone, so a reason code such as "you paid the minimum five times, not six" would make no sense). This was decided after a first run in which the count of months paying the minimum entered the card; including the three counts gives a cross-validated AUC of {{var_month_counts_auc}} against {{var_limit_06_auc}} without them ({{month_counts_gain:+.4f}}), {{month_counts_verdict}}.

## 5. The scorecard

**Weight of evidence.** A feature is cut into bins. Each bin gets WoE = ln(its share of all bad accounts ÷ its share of all good accounts): positive where the bin is riskier than average, negative where safer. The information value (IV) of a feature adds up (share of bad − share of good) × WoE over its bins; under 0.02 is useless, 0.1 to 0.3 medium, above 0.3 strong. Half an account is added to each bin so that a bin with no bad accounts still has a finite weight.

**Binning** ([`creditrisk/binning.py`](../creditrisk/binning.py)) starts with up to 30 small bins and repeatedly merges the neighbouring pair that loses least information, until every bin holds at least 5% of the accounts, there are at most 6 bins, a feature meant to be monotone is monotone, and no neighbouring pair could be merged for a loss of less than 0.002. Cut points are rounded to two significant figures of the feature's spread so that the card reads "45,000", not "44,873". Repayment statuses are binned in their natural order (-2, -1, 0, 1, ...) without forcing the risk to rise, because paying in full is not safer than paying the minimum in this file.

**Selection** ([`creditrisk/scorecard.py`](../creditrisk/scorecard.py)), on the development accounts only:

1. {{n_candidates}} candidate features are binned.
2. {{n_dropped_iv}} are dropped for an information value under 0.02 or for having a single bin.
3. Of two features whose weights of evidence correlate above 0.60, only the one with the higher IV is kept ({{n_dropped_correlated}} dropped). The limit was chosen by cross-validation on the development accounts: 0.4 gives {{var_limit_04_features}} features and a cross-validated AUC of {{var_limit_04_auc}}, 0.5 gives {{var_limit_05_features}} and {{var_limit_05_auc}}, 0.6 gives {{var_limit_06_features}} and {{var_limit_06_auc}}, and 0.7 gives {{var_limit_07_features}} and {{var_limit_07_auc}}, ([`scorecard_variants.csv`](../data/processed/scorecard_variants.csv)). {{limit_choice}}.
4. A logistic regression is fitted to the weights of evidence, and features are dropped one at a time if their coefficient is not positive (a riskier-looking bin must never make an account look safer) or not significant at 5% ({{n_dropped_regression}} dropped).

{{n_features}} features remain: {{features_list}}. Everything dropped, with the reason, is in [`scorecard_screening.csv`](../data/processed/scorecard_screening.csv):

{{screening_table}}

**Points.** The score is `offset + factor × ln(odds of good against bad)`, with factor = {{pdo}} ÷ ln 2 so that every {{pdo}} points double the odds, and the offset chosen so that {{base_score}} points mean odds of {{base_odds}} to 1 (a {{base_odds_pd}} probability of default). Each bin's points are `−factor × (coefficient × WoE + intercept ÷ number of features) + offset ÷ number of features`, rounded to whole points. The probability of default of an account is read from its whole-number score, so the card and the probability always agree; rounding changed the hold-out AUC by {{rounding_change:+.4f}}. Scores on the hold-out run from {{score_min}} to {{score_max}}.

**The card.**

{{scorecard_table}}

## 6. The gradient-boosting challenger

LightGBM on the raw behaviour columns and the engineered measures (no personal details). {{tuning_trials}} settings (leaves, minimum leaf size, learning rate, trees, column and row sampling, regularisation) are drawn at random, always including a sensible default, and the one with the lowest cross-validated log loss on the development accounts is kept ([`challenger_tuning.csv`](../data/processed/challenger_tuning.csv)). {{platt_sentence}} Probabilities are rounded to 0.1 of a percentage point so that the cut-off tables stay small. SHAP values (LightGBM's built-in tree contributions) show what it relies on:

{{importance_table}}

## 7. Validation

All on the hold-out unless stated. **AUC**: the chance that a random defaulter scores riskier than a random payer. **Gini** = 2 × AUC − 1. **KS**: the largest gap between the cumulative score distributions of good and bad accounts. **Brier score**: mean squared error of the probabilities. **Calibration slope**: the slope of a logistic regression of the outcome on the log-odds of the prediction (1.00 is perfect; below 1 means too extreme). **Lift and capture**: default rate by tenth of the riskiest accounts and the share of all defaults found. **PSI**: population stability index of development against hold-out scores. **Bootstrap ranges**: {{bootstrap_draws:,}} resamples of the hold-out accounts, with the same accounts drawn for every model so that differences are paired.

{{metrics_table}}

{{comparisons_table}}

{{gap_sentence}}. {{demographics_sentence}}.

**The hard half.** A rule that reads only last month's status has nothing to go on for accounts that were not late, which is where {{defaults_not_late_share}} of the defaults are:

{{subsets_table}}

**Stability.** The population stability index of the scorecard score is {{psi_score:.3f}}, and the largest for any feature is {{psi_max:.3f}}. The split is random, so this only shows nothing is broken; there is no later period in the file, so stability over time cannot be tested.

{{risk_bands_table}}

## 8. Fairness audit

Sex, age, education and marital status are not inputs to the scorecard or the challenger. They are used only to look at what the decisions do. At the cut-off, for each group: the **approval rate**; the **disparate-impact ratio** (approval rate over that of the best-treated group; under 0.80 is flagged, the "four-fifths" screening rule, which is not a Pakistani legal test); the share of **defaulters declined** and of **good accounts declined**; and **calibration** (average predicted probability over the observed default rate). Groups under {{min_group}} hold-out accounts are shown but not judged. {{fairness_headline}}

{{fairness_table}}

Could the inputs act as a stand-in? A model that tries to tell the groups apart from the scorecard's inputs reaches an AUC of {{proxy_sex:.2f}} for sex, {{proxy_age:.2f}} for being 40 or over and {{proxy_edu:.2f}} for a university education (0.5 means the inputs say nothing about it). A model can be calibrated for every group and still approve groups at different rates when their true default rates differ; the audit shows both so that the two are not confused. This is a screening exercise, not legal advice.

## 9. The economics

Over one billing month, per unit of balance: an account that **pays** earns `margin = (yield − funding − running cost) ÷ 12`, and an account that **defaults** loses `loss`. The **yield** is the card APR times the share of balances that revolve; the APR is {{apr}} (MCB's published retail markup, 3.67% a month, "an APR of up to 44%", July to December 2026, [source]({{mcb_url}})), and {{revolve_pct}} of balances are left unpaid after the next month's payment in this file (measured on the development accounts, over the five month pairs), so the yield is {{yield_pct}}. The **funding cost** is the policy rate ({{funding_pct}} on {{policy_date}}): money lent on a card is money not put in government paper. **Running cost** ({{opex_pct}} a year) and **loss** ({{loss_pct}} of the balance) are assumptions. The margin is {{margin_month_pct}} a month ({{margin_year_pct}} a year).

An account with probability of default p is worth `(1 − p) × margin − p × loss` per unit of balance, which is positive only while `p < margin ÷ (margin + loss)`: **{{break_even}}**. A bad account costs {{cost_ratio:.1f}} good accounts' monthly profit. With calibrated probabilities, approving every account at or below the break-even probability maximises expected profit, whatever the balances. The check on the data: the cut-off that earns most on the development accounts (out-of-fold predictions) is {{empirical_cutoff_pct}}, {{emp_phrase}} the break-even probability of {{break_even}}; it is the break-even probability that is reported, because it does not depend on a noisy peak.

Balances are the September statement, never below zero. Profit is on the hold-out accounts' real outcomes, per account in the whole portfolio (declined accounts earn nothing), with bootstrap ranges:

{{strategies_table}}

**Policy-rate scenarios.** The same calculation with the funding cost set to the policy rate on six dates, and with the first date's cut-off kept and tried under each later date's costs:

{{scenarios_table}}

**Sensitivity.** Share of accounts worth approving and the break-even probability, for combinations of APR and loss on a default, at the current policy rate:

{{sensitivity_table}}

## 10. Limits of the method

- The cardholders are not Pakistani and the book is from a crisis period; nothing here measures how Pakistani cardholders behave.
- "Default" is a missed payment. The loss on a default, the running cost and (to some degree) the APR that applies are assumptions.
- One cohort: no out-of-time test, no view of behaviour through a cycle.
- Selection: only accounts the bank had approved are in the data.
- One month of margin is counted, which understates the value of an account that keeps paying.
- The fairness audit looks at four groups on the information in the file; it does not cover other groups or other notions of fairness, and it is not legal advice.
- The statistical ranges describe sampling noise in a {{n_hold:,}}-account test, not uncertainty about the assumptions.

## 11. Reproducibility

A fixed seed (42) drives the split, the cross-validation, the tuning and the bootstrap. Library versions are pinned in `requirements.txt`. Raw files are checked against `data/raw/MANIFEST.json` by `python -m creditrisk verify`. The tests check each function on made-up portfolios with known answers (weight of evidence, points, the break-even cut-off), run the whole analysis on a sample, and check that the committed results, documents, dashboard and notebook agree with each other.
