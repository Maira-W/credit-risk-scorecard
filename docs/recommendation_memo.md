# Where to draw the approval line on card accounts

*A memo for anyone deciding which cardholders a lender should keep extending credit to. It rests on 30,000 accounts of a Taiwanese bank from 2005, not on Pakistani borrowers, and on stated assumptions about pricing and losses. It shows how the decision would be made; it is not lending advice.*

## Recommendation

- **Score every account with the 8-feature points scorecard and approve only those scoring 549 or more**, which under the base-case assumptions means a probability of default under 10.8%. On the 6,000 hold-out accounts that approves 28% of them; 7.6% of the approved accounts defaulted the next month, against 22.1% of all accounts.
- **Keep the gradient-boosting model as a challenger, not the decision-maker.** Its AUC is 0.792 against 0.768 (real but small) and at the same break-even cut-off it earns NT$104 more per account a month than the scorecard (95% range NT$25 to NT$188; a difference larger than the noise). But the scorecard keeps 92% of its lead over chance, can be read and explained line by line, and gives a reason for every decline. If a regulator or committee is content with a model it cannot read, that trade-off is worth revisiting.
- **Re-set the cut-off whenever the policy rate moves.** The break-even probability was 12.9% at 7.25% (1 October 2021) and 6.2% at 22% (27 June 2023). At the peak, keeping the earlier cut-off would have lost money on the accounts it approved.
- **Leave sex, age, education and marital status out of the model, and audit decisions by those groups every time the cut-off changes.** They are not inputs now, and leaving them out costs almost no accuracy.

## Why

**The risk is in recent behaviour, but 48% of the defaults come from accounts that were not late.** 22.1% of accounts missed the next payment. Last month's repayment status is the strongest signal: 12.8% of accounts that paid the minimum defaulted, 34% of those one month late and 69% of those two months late. On those 4,650 hold-out accounts a rule that reads only last month's status has an AUC of 0.53 (0.5 is a coin toss); the scorecard reaches 0.66, by combining how much of each bill is paid, how much credit is used, the credit limit and the history of the other months.

**How good the scorecard is.** Hold-out AUC 0.768 (95% range 0.752 to 0.783); the riskiest three tenths of accounts hold 61% of all defaults, against 57% for the naive rule (in the riskiest tenth alone all the models find about a third: the obvious cases). Its calibration slope is 1.00 (1.00 is perfect). The development and hold-out score distributions are alike (population stability index 0.002; under 0.10 is stable), which only shows nothing is broken: the file has no later period to test against.

| Hold-out account | Score | Probability of default | Defaulted? | Reasons (points lost against the best bin) |
|---|---:|---:|---|---|
| Riskier than 90% of accounts | 415 | 55.2% | yes | Repayment status, September: 2 or more (-121); Total paid, six months: 14,301 to 22,400 (-21); Credit used, six-month average: over 0.503 up to 0.802 (-15) |
| In the middle | 529 | 14.6% | no | Total paid, six months: 7,100 or less (-37); Credit limit: 70,001 to 140,000 (-11); Credit used, six-month average: over 0.287 up to 0.503 (-9) |
| Safer than 90% of accounts | 567 | 8.1% | no | Total paid, six months: 14,301 to 22,400 (-21); Payment, September: 1 to 3,900 (-7); Share of the previous bill paid, September: over 0 up to 0.234 (-1) |

**What approving is worth depends on three assumptions.** Over one billing month, an approved account that pays earns a margin on its balance and one that defaults loses a share of it. With a card APR of 44% (MCB's published markup is "up to 44%"), 88% of balances left unpaid after the next payment (measured in this file), money costing 12% (the policy rate on 27 October 2026) and running costs of 5%, the margin is 1.82% a month. A default is assumed to cost 15% of the balance, so a bad account costs about 8.2 good accounts' monthly profit, and approving pays only while the probability of default is under 1.82% ÷ (1.82% + 15%) = 10.8%.

| Rule | Approved | Default rate of approved | Profit per account a month | 95% range | Better than approving everyone by |
| --- | ---: | ---: | ---: | ---: | ---: |
| Approve everyone | 100% | 22.1% | -NT$807 | -NT$972 to -NT$658 |  |
| Approve if not late last month (status 0 or better) | 78% | 14.2% | -NT$70 | -NT$193 to NT$43 | NT$738 (NT$643 to NT$842) |
| Scorecard, at the break-even probability | 28% | 7.6% | NT$42 | -NT$57 to NT$138 | NT$850 (NT$730 to NT$986) |
| Gradient boosting, at the break-even probability | 34% | 6.8% | NT$146 | NT$81 to NT$209 | NT$953 (NT$815 to NT$1,103) |

Approving everyone loses NT$807 per account a month in this book; the scorecard at the break-even cut-off earns NT$42. The scorecard's profit is positive on average, but its 95% range reaches below zero, so it is not distinguishable from break-even on 6,000 accounts, and the gain over approving everyone is larger than the noise. Most of the value is losses avoided on the accounts that are declined, not margin earned on those approved.

## The policy rate and the cut-off

| Policy rate on | Rate | Margin a month | Break-even probability | Approved | Default rate of approved | Profit per account a month | If the Oct 2021 cut-off were kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 October 2021 | 7.25% | 2.21% | 12.9% | 42% | 9.1% | NT$156 | NT$156 |
| 8 July 2022 | 15% | 1.57% | 9.5% | 18% | 6.5% | -NT$11 | -NT$28 |
| 27 June 2023 | 22% | 0.99% | 6.2% | 1% | 5.1% | NT$4 | -NT$194 |
| 17 December 2024 | 13% | 1.74% | 10.4% | 25% | 7.5% | NT$7 | NT$20 |
| 16 December 2025 | 10.5% | 1.94% | 11.5% | 33% | 7.8% | NT$85 | NT$79 |
| 27 October 2026 | 12% | 1.82% | 10.8% | 28% | 7.6% | NT$42 | NT$43 |

At 22% the margin on a card balance is thin, so only the safest accounts are worth holding (1% of the book); at 7.25% 42% are. At the peak, keeping the earlier cut-off would have lost money on the accounts it approved (-NT$194 per account a month, against NT$4 with the cut-off reset).

## What could be wrong

- **The loss assumption is the weakest.** "Default" in the data means a missed payment, not a write-off. If a default costs less than assumed the cut-off loosens; if more, it tightens. Across the grid in the [README](../README.md) (APR 30% to 44%, loss 10% to 40%) the break-even probability runs from 1.9% to 15.4%, and the share worth approving from 0% to 55%.
- **The accounts are not Pakistani and are from a crisis-era book.** A lender would refit the scorecard on its own accounts and test it on a later period. The data have only one.
- **Fairness.** At the break-even cut-off, 4 of 11 groups are approved at under four-fifths of the best-treated group's rate: age: 21-29, age: 50 and over, education: High school and education: University; for age: 21-29, age: 50 and over, education: High school and education: University the whole 95% bootstrap range of the ratio is under 0.80, for the rest it reaches above it. Women default less than men in this file (20.8% against 24.2%) and are approved more often (30% against 26%); the model is calibrated for both (predicted ÷ observed 1.00 for women, 0.98 for men). The flagged groups have higher default rates in this file than the best-treated group of the same attribute, and the model is calibrated for them (predicted ÷ observed between 0.93 and 1.02), so the gaps in approval mostly follow gaps in observed default rates, not a model that is harsher than the data on those groups. The largest calibration gap among the groups judged is 7% (age: 50 and over). None of this makes the gaps acceptable: they are the reason to review features and cut-off by group at every re-set.
- **Selection.** These are accounts the bank had approved, so the model has never seen applicants it would have rejected.

## Next steps for a real lender

1. Refit the scorecard on the lender's own accounts and test it on a later period, not a random hold-out.
2. Replace the loss assumption with measured roll rates to write-off and recoveries.
3. Run the new cut-off alongside the old one on a small share of accounts before switching.
4. Monitor the score distribution monthly (population stability index above 0.10 is a warning, above 0.25 a reason to refit), and the default rate of each score band against its prediction.
5. Review the audit by group at each re-set of the cut-off, and keep the reason codes with every decline.
