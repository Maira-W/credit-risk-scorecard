# Credit card default scorecard

**Who will miss next month's card payment, and where should a lender draw the line?**

A points scorecard of the kind banks use, built on {{accounts:,}} credit card accounts, tested against a gradient-boosting model, checked for calibration and fairness, and priced with Pakistan's policy rate: the dearer money gets, the safer an account has to be to be worth approving.

[**Interactive dashboard**](dashboard/dashboard.html) (open it in a browser; no server needed) · [Recommendation memo](docs/recommendation_memo.md) · [Method](docs/methodology.md) · [Model card](docs/model_card.md) · [Data sources](docs/data_sources.md) · [Walkthrough notebook](notebooks/credit_risk_walkthrough.ipynb)

> **Read this first.** The cardholders are {{accounts:,}} accounts of one bank in Taiwan in 2005, not Pakistani borrowers (I found no public borrower-level credit data for Pakistan). "Default" means *missing next month's payment*, not a loan written off. What is Pakistani is the cost of money (the State Bank of Pakistan's policy rate, {{rate_now}} on {{rate_now_date}}) and the card pricing (MCB's published markup). The running cost is an assumption and the loss on a missed payment is estimated from this file's own roll rates, with two assumptions of its own; the answer moves a lot with them. This is a method demonstration, not lending advice.

## The answer

- **The risk is in recent behaviour.** {{default_rate_pct}} of the accounts missed the next payment. Accounts that paid the minimum last month defaulted {{rate_status_ok}} of the time, accounts one month late {{rate_status_one}}, two months late {{rate_status_two}}. The {{late_share}} of accounts that were late last month produced {{defaults_late_share}} of the defaults, so {{defaults_not_late_share}} of the defaults came from accounts that looked fine on that measure.
- **{{h_compare}}** On {{n_hold:,}} hold-out accounts the scorecard has an AUC of **{{auc_scorecard:.3f}}** (95% range {{auc_scorecard_lo:.3f}} to {{auc_scorecard_hi:.3f}}): a random defaulter scores riskier than a random payer {{auc_scorecard:.1%}} of the time. A rule that only reads last month's status gets {{auc_naive:.3f}}; gradient boosting gets {{auc_challenger:.3f}}. {{gap_sentence}}. The scorecard keeps {{share_of_lift}} of the gradient-boosting model's lead over a coin toss, and unlike it can be read line by line.
- **{{h_hard}}** Among the {{hard_accounts:,}} hold-out accounts that were *not* late last month ({{hard_defaults}} of them defaulted), the AUC is {{hard_auc_naive:.2f}} for the naive rule, {{hard_auc_scorecard:.2f}} for the scorecard and {{hard_auc_challenger:.2f}} for gradient boosting (0.5 is a coin toss).
- **{{h_calibration}}** Calibration slope {{slope_scorecard:.2f}} on the hold-out (1.00 is perfect). The riskiest three tenths of accounts hold {{capture3_scorecard}} of all defaults, against {{capture3_naive}} for the naive rule and {{capture3_challenger}} for gradient boosting; in the riskiest tenth alone all three find about a third ({{capture_scorecard}}, {{capture_naive}}, {{capture_challenger}}): the obvious cases.
- **Fairness.** Sex, age, education and marital status are not inputs. {{fairness_headline}} {{demographics_sentence}}.
- **The cut-off depends on the cost of money.** Under the stated assumptions (card APR {{apr}}, money at the policy rate of {{funding_pct}}, loss {{loss_pct}} of the balance on a missed payment, estimated from the roll rates) an account is worth approving only while its probability of default is under **{{break_even}}**, which is a score of **{{cutoff_score}}** or more. On the hold-out accounts that approves {{approved_share}} of them, {{bad_rate_approved}} of whom default, and earns {{profit_scorecard}} per account a month (95% range {{profit_scorecard_lo}} to {{profit_scorecard_hi}}) against {{profit_all}} if everyone were approved. {{scorecard_profit_verdict}}, {{gain_verdict}}. At the same cut-off gradient boosting earns {{profit_challenger}} ({{ch_vs_sc_phrase}}; 95% range {{ch_vs_sc_lo}} to {{ch_vs_sc_hi}}: {{ch_vs_sc_verdict}}).
- **When the policy rate moves, so should the cut-off.** At the {{scen_first_rate}} low of {{scen_first_date}} the break-even probability is {{scen_first_be}} and {{scen_first_share}} of accounts are worth approving; at the {{scen_peak_rate}} peak of {{scen_peak_date}} it is {{scen_peak_be}} and {{scen_peak_share}}. At the peak, {{scen_peak_verdict}} ({{scen_peak_kept}} per account a month, against {{scen_peak_profit}} with the cut-off reset). {{scen_note}}
- **The loss on a missed payment is estimated from the data, not guessed.** In the file's own repayment history an account that is late recovers by the next month {{roll_cure}} of the time, stays at the same level {{roll_stay}} and gets worse {{roll_worse}}. The chance that an account that misses a payment is written off within {{horizon_months}} months is {{p_wo_base}} (95% range {{p_wo_base_range}}) at the conservative status-{{writeoff_status}} line, which with {{lgd_pct}} of the balance never recovered and {{collection_pct}} for collection is a loss of {{loss_base}}. {{loss_cases_sentence}}
- **An explainable boosting model.** {{mono_sentence}}
- **Stress tests and pricing.** {{stress_sentence}} {{pricing_sentence}}

## Figures

| | |
|---|---|
| ![Default rate by last month's repayment status](docs/figures/01_default_by_status.png) | ![Accuracy of the models with 95% ranges](docs/figures/04_model_comparison.png) |
| ![Calibration](docs/figures/05_calibration.png) | ![The scorecard's points](docs/figures/08_scorecard_points.png) |
| ![Profit against share approved](docs/figures/10_profit_curve.png) | ![Policy-rate scenarios](docs/figures/11_policy_rate_scenarios.png) |

The second layer (roll rates and the loss, explainable boosting, swap sets, stress tests and pricing, what drives the fairness gaps):

| | |
|---|---|
| ![Roll rates](docs/figures/15_roll_rates.png) | ![What each loss does to the cut-off](docs/figures/16_loss_cases.png) |
| ![Explainable boosting](docs/figures/17_explainable_boosting.png) | ![Swap sets](docs/figures/18_swap_sets.png) |
| ![Stress tests and pricing](docs/figures/19_stress_and_pricing.png) | ![What drives the approval gaps](docs/figures/20_fairness_drivers.png) |

More in [`docs/figures`](docs/figures): risk drivers, ROC and lift curves, score distribution, what the gradient-boosting model relies on, the sensitivity grid, the fairness audit and the Pakistan context.

## What was done

1. **Data.** The UCI "Default of Credit Card Clients" file ({{accounts:,}} accounts, six months of status, bills and payments, and whether the account missed its payment in October 2005), cleaned without dropping any account ([details](docs/data_sources.md)).
2. **Split.** {{n_dev:,}} accounts for building everything, {{n_hold:,}} kept aside and used once. Identical accounts never straddle the split.
3. **Features.** {{n_engineered}} measures built from each account's own history (months late, share of the last bill paid, credit used, and so on).
4. **Scorecard.** Each candidate feature is binned by merging neighbours that lose least information, weights of evidence are fed to a logistic regression with positive, significant coefficients, and the result becomes whole points: {{base_score}} points at {{base_odds}}:1 odds of a good account, {{pdo}} points to double the odds. {{n_features}} of {{n_candidates}} candidates ended up in the card, in {{n_bins}} bins.
5. **Challenger.** LightGBM, {{tuning_trials}} settings tried by cross-validation, recalibrated; plus a naive rule for scale.
6. **Validation.** AUC, KS, Brier score, calibration, lift and population stability, with bootstrap ranges; cross-validation with every step refitted.
7. **Fairness audit.** Approval, error rates and calibration by sex, age, education and marital status, and whether the inputs act as a stand-in for them.
8. **Cut-off.** Profit per unit of balance, month by month, under stated assumptions, then repeated for the policy rate on six dates and for a grid of assumptions.
9. **Going further.** The loss on a missed payment measured from roll rates instead of guessed, an explainable boosting model whose directions are checked, swap sets between models, stress tests, risk-based pricing, re-anchoring the scorecard to another lender, and what drives the approval gaps (the section below).

## Results

{{metrics_table}}

AUC is the chance that a random defaulter scores riskier than a random payer (0.5 is a coin toss, 1 is perfect); KS is the largest gap between the two groups' cumulative score distributions. Cross-validated AUC is the average of {{cv_repeats}} × {{cv_folds}} refits on the development accounts; the hold-out columns are the single final test. Which differences are bigger than chance:

{{comparisons_table}}

### The scorecard

The card's three heaviest features are {{top_feature_1}} ({{top_range_1}} points between its best and worst bin), {{top_feature_2}} ({{top_range_2}}) and {{top_feature_3}} ({{top_range_3}}). The score is the sum of the points; more points is safer. The full card is in the [method](docs/methodology.md); the first three features:

{{scorecard_top_table}}

The gradient-boosting model leans on {{challenger_top}}.

### What approving is worth

Profit per account in the whole portfolio, per month, on the hold-out accounts and the base-case assumptions (declined accounts earn nothing):

{{strategies_table}}

### The policy rate and the cut-off

{{scenarios_table}}

### How much the assumptions matter

Share of accounts worth approving (and the break-even probability) for combinations of card APR and loss on a missed payment (the three roll-rate values, then {{loss_stress_pct}} and 30% as stresses), at the current policy rate; across this grid the break-even probability runs from {{sens_be_min}} to {{sens_be_max}} and the share approved from {{sens_share_min}} to {{sens_share_max}}:

{{sensitivity_table}}

### Fairness at the cut-off

{{fairness_table}}

"Disparate impact" is a group's approval rate over that of the best-treated group; under 0.80 is flagged (the "four-fifths" screening rule used in US lending and employment practice, not a Pakistani legal test). "Predicted ÷ observed" near 1.00 means the model is neither too harsh nor too lenient on that group on average. Details and the proxy test are in the [model card](docs/model_card.md).

## Going further

### What a missed payment costs: roll rates

The label is a missed payment, not a write-off, so the loss on it is not in the data. It is estimated from how accounts move between repayment statuses from one month to the next: {{roll_moves}} month-to-month moves in {{roll_months}}, the months coded on one basis. A late account recovers by the next month {{roll_cure}} of the time, stays at the same level {{roll_stay}} and gets worse {{roll_worse}}; {{roll_into_late}} of current accounts become late in a month.

{{rollrates_matrix_table}}

The accounts that really missed their payment stood in different places in September ({{entry_current}} were current, {{entry_first_late}} at the first late level, {{entry_deeper}} deeper). Having missed a payment, each is late in October: a current account has just slipped to the first late level, a late one stayed or got worse. So each is moved one month on using the matrix's own odds of the late states ({{landing_first}} land on the first late level, {{landing_second}} one level deeper, {{landing_line}} at or beyond the write-off line), and the ladder is followed from there for {{horizon_months}} months. That gives the chance of a write-off, and so the loss:

{{evidence_table}}

Loss = {{lgd_pct}} of the balance never recovered x the chance + {{collection_pct}} for collection and the income lost while the account is late. {{split_sentence}}

{{split_table}}

The base case uses the conservative line, status {{writeoff_status}}; {{loss_stress_pct}} (the placeholder used before the roll rates were estimated) is kept as a stress case. The ladder assumes the roll rates of {{roll_months}} hold for a year, in a file that is the start of a card-debt crisis, so later roll rates were probably worse: read the loss as a floor, not a forecast. How the status codes map to months is not documented.

What each loss does to the cut-off (hold-out accounts, base-case yield and funding cost):

{{loss_cases_table}}

### An explainable boosting model

The gradient-boosting model is told, for {{mono_constraints}} features where the direction is obvious (credit limit, payments, months late, the share of the bill paid), that more of the feature can never mean less risk. Repayment statuses and credit used are left free, because the first are codes and the second is U-shaped at the low end.

{{advanced_models_table}}

{{mono_sentence}}

### Swap sets

A swap set shows where two models disagree and who was right. At the break-even cut-off:

{{swap_table}}

{{swap_sentence}}

### Stress tests and pricing

Expected profit per account a month from the scorecard's own probabilities, when defaults rise (the odds are multiplied by the factor in the name) or money gets dearer, with the cut-off kept and with it re-set. These are expected values, not outcomes: the file has no accounts that lived through a different year.

{{stress_table}}

{{stress_sentence}}

Risk-based pricing asks the reverse question: what APR would make approving each tenth of accounts break even?

{{pricing_table}}

{{pricing_sentence}}

### Moving the scorecard to another lender

The ranking carries over to another lender's accounts; the probabilities do not. A lender whose accounts miss a payment at a different rate moves every log-odds by the difference between the two base log-odds (the usual correction for a change in the base rate) and gets a different cut-off:

{{local_rate_table}}

{{local_sentence}}

### What drives the approval gaps

The scorecard adds up points, so a group's gap to the best-treated group of its attribute splits by feature. The table covers the groups that are flagged or within 10 points of the four-fifths line (disparate impact under {{watch_below}}). {{drivers_sentence}}

{{drivers_table}}

## What this cannot tell you

- **Not Pakistani data.** The accounts are Taiwanese, from a crisis-era card book where {{default_rate_pct}} missed a payment in one month. A Pakistani lender would refit the scorecard on its own accounts; what carries over is the method and the link between the policy rate and the cut-off.
- **Default is a missed payment.** It is not a charge-off, so the loss on it is estimated from roll rates measured over only {{roll_months}}, at the start of a crisis (a floor, not a forecast), and the cut-off moves a lot with it.
- **The economics are assumptions.** Only the card APR ({{apr}}, MCB's published "up to") and the policy rate are sourced. The share of balances that revolve ({{revolve_pct}}) and the roll rates behind the loss ({{p_wo_base}} of missed payments reach the write-off line within a year) are measured in this file; running cost ({{opex_pct}}), the share of a written-off balance never recovered ({{lgd_pct}}) and the cost of collection ({{collection_pct}}) are assumptions, and together they make the loss {{loss_pct}}.
- **One cohort.** The file is a single snapshot, so there is no later period to test the scorecard against and no way to see how it holds up through a cycle.
- **Selection.** These are accounts the bank had already approved. Applicants it turned down are not here.
- **Repayment codes.** The documentation does not explain a status of -2 or 0, and a status of 1 ("one month late") appears {{status_one_sep:,}} times in September but only {{status_one_rest}} times in the four months before August, so the earlier months look to be coded on another basis. The scorecard treats any status of 1 or more as late.
- **One month of margin.** An approved account that keeps paying earns again next month; counting one month understates what good accounts are worth.

## Run it

```
pip install -r requirements.txt
python -m creditrisk verify      # every raw file matches its checksum
python -m creditrisk build       # clean the data and make the features
python -m creditrisk analyze     # fit, validate, audit, price (about 10 minutes)
python -m creditrisk report      # figures, dashboard and documents
python -m creditrisk all         # build, analyze and report in one go
```

The raw data are in the repository (they are CC BY 4.0), so nothing is downloaded unless you ask for it with `python -m creditrisk ingest --refresh`. The notebook is rebuilt with `pip install -r requirements-notebook.txt` and `python -m creditrisk notebook`. Tests and the linter: `pip install -r requirements-dev.txt`, then `pytest -q` and `ruff check .`. A fixed seed makes a rerun on the same versions of the libraries give the same numbers.

## Repository map

| Path | What is in it |
|---|---|
| `creditrisk/` | the package: `ingest`, `build`, `binning`, `scorecard`, `models`, `validation`, `fairness`, `economics`, `rollrates`, `advanced`, `macro`, `analysis`, `charts`, `dashboard`, `docs`, `notebook` |
| `data/raw/` | the UCI file and the World Bank series, with `MANIFEST.json` (source, date, size, SHA-256 of each) |
| `data/reference/` | the SBP policy rate, typed in from announcements with a source for every change |
| `data/processed/` | every table behind the figures and documents, and `summary.json` |
| `docs/` | the memo, method, model card, data sources and the figures |
| `dashboard/` | the interactive dashboard: one HTML file with the data inside |
| `notebooks/` | a walkthrough that refits the scorecard and checks it against the committed results |
| `tests/` | automated tests on made-up portfolios with known answers, and on the committed results |

## Data, licences and credit

Cardholder data: {{citation}} UCI Machine Learning Repository, [dataset 350]({{uci_page}}), CC BY 4.0. Pakistan bank and credit series: World Bank, CC BY 4.0. Card APR: [MCB credit card summary box]({{mcb_url}}), July to December 2026. Policy rates: State Bank of Pakistan announcements as reported (a source for each change is in [`data/reference/policy_rate.csv`](data/reference/policy_rate.csv)). The code and documents are released under the [MIT licence](LICENSE); the data belong to their sources, see [`docs/data_sources.md`](docs/data_sources.md).
