# Where to draw the approval line on card accounts

*A memo for anyone deciding which cardholders a lender should keep extending credit to. It rests on {{accounts:,}} accounts of a Taiwanese bank from 2005, not on Pakistani borrowers, and on stated assumptions about pricing and losses. It shows how the decision would be made; it is not lending advice.*

## Recommendation

- **Score every account with the {{n_features}}-feature points scorecard and approve only those scoring {{cutoff_score}} or more**, which under the base-case assumptions means a probability of default under {{break_even}}. On the {{n_hold:,}} hold-out accounts that approves {{approved_share}} of them; {{bad_rate_approved}} of the approved accounts defaulted the next month, against {{default_rate_hold_pct}} of all accounts. {{scorecard_profit_verdict}} (see "Why" below).
- **Keep the gradient-boosting model as a challenger, not the decision-maker.** Its AUC is {{auc_challenger:.3f}} against {{auc_scorecard:.3f}} ({{gap_verdict}}) and at the same break-even cut-off it earns {{ch_vs_sc_phrase}} (95% range {{ch_vs_sc_lo}} to {{ch_vs_sc_hi}}; {{ch_vs_sc_verdict}}). But the scorecard keeps {{share_of_lift}} of its lead over chance, can be read and explained line by line, and gives a reason for every decline. If a regulator or committee is content with a model it cannot read, that trade-off is worth revisiting.
- **Re-set the cut-off whenever the policy rate moves.** The break-even probability was {{scen_first_be}} at {{scen_first_rate}} ({{scen_first_date}}) and {{scen_peak_be}} at {{scen_peak_rate}} ({{scen_peak_date}}). At the peak, {{scen_peak_verdict}}.
- **Leave sex, age, education and marital status out of the model, and audit decisions by those groups every time the cut-off changes.** They are not inputs now, and leaving them out costs {{demo_cost}}.

## Why

**The risk is in recent behaviour, but {{defaults_not_late_share}} of the defaults come from accounts that were not late.** {{default_rate_pct}} of accounts missed the next payment. Last month's repayment status is the strongest signal: {{rate_status_ok}} of accounts that paid the minimum defaulted, {{rate_status_one}} of those one month late and {{rate_status_two}} of those two months late. On those {{hard_accounts:,}} hold-out accounts a rule that reads only last month's status has an AUC of {{hard_auc_naive:.2f}} (0.5 is a coin toss); the scorecard reaches {{hard_auc_scorecard:.2f}}, by combining how much of each bill is paid, how much credit is used, the credit limit and the history of the other months.

**How good the scorecard is.** Hold-out AUC {{auc_scorecard:.3f}} (95% range {{auc_scorecard_lo:.3f}} to {{auc_scorecard_hi:.3f}}); the riskiest three tenths of accounts hold {{capture3_scorecard}} of all defaults, against {{capture3_naive}} for the naive rule (in the riskiest tenth alone all the models find about a third: the obvious cases). Its calibration slope is {{slope_scorecard:.2f}} (1.00 is perfect). The development and hold-out score distributions are alike (population stability index {{psi_score:.3f}}; under 0.10 is stable), which only shows nothing is broken: the file has no later period to test against.

| Hold-out account | Score | Probability of default | Defaulted? | Reasons (points lost against the best bin) |
|---|---:|---:|---|---|
{{examples_rows}}

**What approving is worth depends on three assumptions.** Over one billing month, an approved account that pays earns a margin on its balance and one that defaults loses a share of it. With a card APR of {{apr}} (MCB's published markup is "up to 44%"), {{revolve_pct}} of balances left unpaid after the next payment (measured in this file), money costing {{funding_pct}} (the policy rate on {{policy_date}}) and running costs of {{opex_pct}}, the margin is {{margin_month_pct}} a month. A default is assumed to cost {{loss_pct}} of the balance, so a bad account costs about {{cost_ratio:.1f}} good accounts' monthly profit, and approving pays only while the probability of default is under {{margin_month_pct}} ÷ ({{margin_month_pct}} + {{loss_pct}}) = {{break_even}}.

{{strategies_table}}

{{all_phrase}}; the scorecard at the break-even cut-off earns {{profit_scorecard}}. {{scorecard_profit_verdict}}, {{gain_verdict}}. Most of the value is losses avoided on the accounts that are declined, not margin earned on those approved.

## The policy rate and the cut-off

{{scenarios_table}}

At {{scen_peak_rate}} the margin on a card balance is thin, so only the safest accounts are worth holding ({{scen_peak_share}} of the book); at {{scen_first_rate}} {{scen_first_share}} are. At the peak, {{scen_peak_verdict}} ({{scen_peak_kept}} per account a month, against {{scen_peak_profit}} with the cut-off reset). {{scen_note}}

## What could be wrong

- **The loss assumption is the weakest.** "Default" in the data means a missed payment, not a write-off. If a default costs less than assumed the cut-off loosens; if more, it tightens. Across the grid in the [README](../README.md) (APR 30% to 44%, loss 10% to 40%) the break-even probability runs from {{sens_be_min}} to {{sens_be_max}}, and the share worth approving from {{sens_share_min}} to {{sens_share_max}}.
- **The accounts are not Pakistani and are from a crisis-era book.** A lender would refit the scorecard on its own accounts and test it on a later period. The data have only one.
- **Fairness.** {{fairness_headline}} {{sex_sentence}} {{fairness_reading}} The largest calibration gap among the groups judged is {{worst_calibration}} ({{worst_calibration_group}}). None of this makes the gaps acceptable: they are the reason to review features and cut-off by group at every re-set.
- **Selection.** These are accounts the bank had approved, so the model has never seen applicants it would have rejected.

## Next steps for a real lender

1. Refit the scorecard on the lender's own accounts and test it on a later period, not a random hold-out.
2. Replace the loss assumption with measured roll rates to write-off and recoveries.
3. Run the new cut-off alongside the old one on a small share of accounts before switching.
4. Monitor the score distribution monthly (population stability index above 0.10 is a warning, above 0.25 a reason to refit), and the default rate of each score band against its prediction.
5. Review the audit by group at each re-set of the cut-off, and keep the reason codes with every decline.
