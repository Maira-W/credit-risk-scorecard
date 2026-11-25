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

Over one billing month, per unit of balance: an account that **pays** earns `margin = (yield − funding − running cost) ÷ 12`, and an account that **defaults** loses `loss`. The **yield** is the card APR times the share of balances that revolve; the APR is {{apr}} (MCB's published retail markup, 3.67% a month, "an APR of up to 44%", July to December 2026, [source]({{mcb_url}})), and {{revolve_pct}} of balances are left unpaid after the next month's payment in this file (measured on the development accounts, over the five month pairs), so the yield is {{yield_pct}}. The **funding cost** is the policy rate ({{funding_pct}} on {{policy_date}}): money lent on a card is money not put in government paper. **Running cost** ({{opex_pct}} a year) is an assumption. The **loss** ({{loss_pct}} of the balance on an account that misses a payment) is measured from the roll rates of the repayment history (section 10), with two assumptions of its own. The margin is {{margin_month_pct}} a month ({{margin_year_pct}} a year).

An account with probability of default p is worth `(1 − p) × margin − p × loss` per unit of balance, which is positive only while `p < margin ÷ (margin + loss)`: **{{break_even}}**. A bad account costs {{cost_ratio:.1f}} good accounts' monthly profit. With calibrated probabilities, approving every account at or below the break-even probability maximises expected profit, whatever the balances. The check on the data: the cut-off that earns most on the development accounts (out-of-fold predictions) is {{empirical_cutoff_pct}}, {{emp_phrase}} the break-even probability of {{break_even}}; it is the break-even probability that is reported, because it does not depend on a noisy peak.

Balances are the September statement, never below zero. Profit is on the hold-out accounts' real outcomes, per account in the whole portfolio (declined accounts earn nothing), with bootstrap ranges:

{{strategies_table}}

**Policy-rate scenarios.** The same calculation with the funding cost set to the policy rate on six dates, and with the first date's cut-off kept and tried under each later date's costs:

{{scenarios_table}}

**Sensitivity.** Share of accounts worth approving and the break-even probability, for combinations of APR and loss on a missed payment, at the current policy rate:

{{sensitivity_table}}

## 10. Roll rates and the loss on a missed payment

The label is a missed payment, so the loss on it cannot be read from the data. [`creditrisk/rollrates.py`](../creditrisk/rollrates.py) estimates it from how accounts move between repayment statuses.

- **States.** The five months {{roll_months}} are the ones coded on one basis (a status of 1 hardly appears in them). A status of 0 or below is "current"; the late statuses 2, 3, 4, 5 and 6-or-more are five further states (a status of 1 joins 2).
- **Transition matrix.** Counting every account's move from one month to the next over the four month pairs gives {{roll_moves}} moves, and from them the chance of going from each state to each other state.

{{rollrates_matrix_table}}

- **From a missed payment to a write-off.** Treating the matrix as steady, raising it to the power of the number of months gives the chance that an account that starts in a given state reaches a given status within {{horizon_months}} months, counting it once it gets there.
- **Where the chain starts.** The label is a missed payment in October, so every account that defaulted is late in October, whatever it was in September ({{entry_current}} of them were current, {{entry_first_late}} at the first late level, {{entry_deeper}} deeper). Starting the chain from September's state would treat the current ones as not having missed yet, and let the late ones cure in the very step the label says they did not. Instead each is moved one month on, using the matrix's own odds of each late state from where it stood with the odds of being current set aside and the rest scaled up to one (`october_entry`): a current account lands on the first late level; a late one stays or gets worse. That puts {{landing_first}} of the missed payments on the first late level, {{landing_second}} one level deeper and {{landing_line}} at or beyond the write-off line, and the chain runs from there. The 95% ranges resample accounts, so they cover the uncertainty of both the matrix and the starting point. The method is checked on simulated accounts that follow a known chain, where the true chance of a write-off after a missed payment can be counted (`tests/test_rollrates.py`).
- **A check against September.** The matrix was estimated on April to August, so the step from September to October is one it has not seen. Accounts at status 2 in September did not default {{sep_two_pays}} of the time, {{sep_two_vs_cure}} the {{roll_cure}} of late accounts that recovered in a month between April and August. Accounts at status 1 did not default {{sep_one_pays}} of the time: that status hardly appears before September, so the matrix has little to say about it. And accounts that were not late in September defaulted {{sep_not_late_default}} of the time, against the {{roll_into_late}} of current accounts that became late in an average month before. Either the crisis deepened in October or September's codes count lateness differently from the earlier months; the data cannot separate the two. Either way the roll rates of April to August probably understate how bad things got, which is one more reason to read the loss as a floor.
- **Loss.** loss = (share never recovered) x (chance of write-off) + (cost of collection and of the income lost while late). The share never recovered ({{lgd_pct}}) and the cost ({{collection_pct}} of the balance) are assumptions; the chance is measured.

{{evidence_table}}

{{split_sentence}}

{{split_table}}

The base case uses the conservative line (status {{writeoff_status}}) because the codes are not documented in months and the history is the start of a crisis, when roll rates were rising. {{loss_stress_pct}}, the placeholder used before this was measured, is kept as a stress case. What each loss does to the cut-off, on the hold-out accounts:

{{loss_cases_table}}

{{loss_cases_sentence}}

## 11. An explainable boosting model

LightGBM can be told that a feature may only push the probability of default one way. The constraints are in `config.MONOTONE`: credit limit, payments and the share of the bill paid can only lower the probability as they rise; months late, recent late months and the worst status can only raise it. Repayment statuses are codes and credit used is U-shaped at the low end, so they are left free. The same settings as the free model are used and the probabilities are recalibrated the same way. Whether the model obeys is checked, not assumed: for a sample of accounts each constrained feature is moved across its range with everything else fixed, and any move of the probability the wrong way is counted.

{{advanced_models_table}}

{{mono_sentence}}

## 12. Swap sets

A swap-set analysis, standard in model validation, compares two models by the accounts on which they disagree. At the break-even cut-off, each account is approved by both models, by one, or by neither; for each group the table shows how many, how many defaulted, their average balance, and what approving them would have earned (realised, per account of the whole portfolio).

{{swap_table}}

{{swap_sentence}}

## 13. Stress tests

A stress test asks what a shock does to profit. Defaults are raised by multiplying every account's odds of default by a factor (for small probabilities this is close to multiplying the probability), money gets dearer by taking the funding cost from the policy rate at its 2023 peak, or both. For each scenario the expected profit per account is worked out from the scorecard's own probabilities for the accounts approved under two policies: the cut-off kept where it is, and the cut-off re-set to the scenario's break-even. These are expected values, not outcomes: the file has no accounts that lived through a different year.

{{stress_table}}

{{stress_sentence}}

## 14. Risk-based pricing

The same arithmetic can be turned round. The margin on a good account must cover the expected loss, p x loss / (1 - p), so the APR that makes approving a group break even is (12 x that margin + funding + running cost) / (share of balances that revolve). For each tenth of accounts, from the safest:

{{pricing_table}}

{{pricing_sentence}}

## 15. Moving the scorecard to another lender

A scorecard fitted on accounts that miss a payment at one rate can be used on a lender whose accounts miss at another. The ranking is kept; every log-odds moves by logit(new rate) - logit(old rate), the usual correction for a change in the base rate (it assumes the way the features relate to risk is the same, which is the part that needs a local test). The break-even cut-off, in the original probabilities, is then the break-even probability moved back by the same amount.

{{local_rate_table}}

{{local_sentence}}

## 16. What drives the approval gaps

The scorecard is a sum of points, so the gap in average score between a group and the best-treated group of its attribute is the sum of the gaps in the points each feature gives. The table covers every group that is flagged or within 10 points of the four-fifths line (disparate impact under {{watch_below}}) and lists the three features that contribute most to its gap.

{{drivers_table}}

{{drivers_sentence}}

## 17. Limits of the method

- The cardholders are not Pakistani and the book is from a crisis period; nothing here measures how Pakistani cardholders behave.
- "Default" is a missed payment. The loss on it comes from roll rates over five months at the start of a crisis, with two assumptions (the share never recovered, the cost of collection); the running cost and (to some degree) the APR that applies are assumptions too. The re-anchoring to another lender assumes the features relate to risk in the same way there.
- One cohort: no out-of-time test, no view of behaviour through a cycle.
- Selection: only accounts the bank had approved are in the data.
- One month of margin is counted, which understates the value of an account that keeps paying.
- The fairness audit looks at four groups on the information in the file; it does not cover other groups or other notions of fairness, and it is not legal advice.
- The statistical ranges describe sampling noise in a {{n_hold:,}}-account test, not uncertainty about the assumptions.

## 18. Reproducibility

A fixed seed (42) drives the split, the cross-validation, the tuning and the bootstrap. Library versions are pinned in `requirements.txt`. Raw files are checked against `data/raw/MANIFEST.json` by `python -m creditrisk verify`. The tests check each function on made-up portfolios with known answers (weight of evidence, points, the break-even cut-off), run the whole analysis on a sample, and check that the committed results, documents, dashboard and notebook agree with each other.
