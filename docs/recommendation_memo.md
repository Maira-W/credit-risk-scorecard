# Where to draw the approval line on card accounts

*A memo for anyone deciding which cardholders a lender should keep extending credit to. It rests on 30,000 accounts of a Taiwanese bank from 2005, not on Pakistani borrowers, and on stated assumptions about pricing and losses. It shows how the decision would be made; it is not lending advice.*

## Recommendation

- **Score every account with the 8-feature points scorecard and approve only those scoring 535 or more**, which under the base-case assumptions means a probability of default under 13.4%. On the 6,000 hold-out accounts that approves 45% of them; 9.2% of the approved accounts defaulted the next month, against 22.1% of all accounts. The scorecard's profit is clearly above zero on this test (see "Why" below).
- **Keep the gradient-boosting model as a challenger, not the decision-maker.** Its AUC is 0.792 against 0.768 (real but small) and at the same break-even cut-off it earns NT$37 more per account a month than the scorecard (95% range -NT$25 to NT$96; a difference that could be chance on a test this size). But the scorecard keeps 92% of its lead over chance, can be read and explained line by line, and gives a reason for every decline. If a regulator or committee is content with a model it cannot read, that trade-off is worth revisiting; a middle path is monotone boosting, forced to respect common-sense directions (AUC 0.792, NT$185 per account a month at the same cut-off).
- **Re-set the cut-off whenever the policy rate moves.** The break-even probability was 15.9% at 7.25% (1 October 2021) and 7.7% at 22% (27 June 2023). At the peak, keeping the earlier cut-off would have lost money on the accounts it approved.
- **Leave sex, age, education and marital status out of the model, and audit decisions by those groups every time the cut-off changes.** They are not inputs now, and leaving them out costs almost no accuracy.

## Why

**The risk is in recent behaviour, but 48% of the defaults come from accounts that were not late.** 22.1% of accounts missed the next payment. Last month's repayment status is the strongest signal: 12.8% of accounts that paid the minimum defaulted, 34% of those one month late and 69% of those two months late. On those 4,650 hold-out accounts a rule that reads only last month's status has an AUC of 0.53 (0.5 is a coin toss); the scorecard reaches 0.66, by combining how much of each bill is paid, how much credit is used, the credit limit and the history of the other months.

**How good the scorecard is.** Hold-out AUC 0.768 (95% range 0.752 to 0.783); the riskiest three tenths of accounts hold 61% of all defaults, against 57% for the naive rule (in the riskiest tenth alone all the models find about a third: the obvious cases). Its calibration slope is 1.00 (1.00 is perfect). The development and hold-out score distributions are alike (population stability index 0.002; under 0.10 is stable), which only shows nothing is broken: the file has no later period to test against.

| Hold-out account | Score | Probability of default | Defaulted? | Reasons (points lost against the best bin) |
|---|---:|---:|---|---|
| Riskier than 90% of accounts | 415 | 55.2% | yes | Repayment status, September: 2 or more (-121); Total paid, six months: 14,301 to 22,400 (-21); Credit used, six-month average: over 0.503 up to 0.802 (-15) |
| In the middle | 529 | 14.6% | no | Total paid, six months: 7,100 or less (-37); Credit limit: 70,001 to 140,000 (-11); Credit used, six-month average: over 0.287 up to 0.503 (-9) |
| Safer than 90% of accounts | 567 | 8.1% | no | Total paid, six months: 14,301 to 22,400 (-21); Payment, September: 1 to 3,900 (-7); Share of the previous bill paid, September: over 0 up to 0.234 (-1) |

**What approving is worth depends on three assumptions.** Over one billing month, an approved account that pays earns a margin on its balance and one that defaults loses a share of it. With a card APR of 44% (MCB's published markup is "up to 44%"), 88% of balances left unpaid after the next payment (measured in this file), money costing 12% (the policy rate on 27 October 2026) and running costs of 5%, the margin is 1.82% a month. The loss on a missed payment is 11.8% of the balance, estimated from how accounts roll towards write-off in the file's own repayment history (11.5% are written off within 12 months at the status-4 line, times 85% never recovered, plus 2% for collection), so a bad account costs about 6.5 good accounts' monthly profit, and approving pays only while the probability of default is under 1.82% ÷ (1.82% + 11.8%) = 13.4%.

| Rule | Approved | Default rate of approved | Profit per account a month | 95% range | Better than approving everyone by |
| --- | ---: | ---: | ---: | ---: | ---: |
| Approve everyone | 100% | 22.1% | -NT$473 | -NT$607 to -NT$354 |  |
| Approve if not late last month (status 0 or better) | 78% | 14.2% | NT$89 | -NT$9 to NT$179 | NT$562 (NT$488 to NT$645) |
| Scorecard, at the break-even probability | 45% | 9.2% | NT$154 | NT$65 to NT$236 | NT$627 (NT$539 to NT$722) |
| Gradient boosting, at the break-even probability | 44% | 7.9% | NT$191 | NT$121 to NT$254 | NT$663 (NT$559 to NT$769) |

Approving everyone loses NT$473 per account a month in this book; the scorecard at the break-even cut-off earns NT$154. The scorecard's profit is clearly above zero on this test, and the gain over approving everyone is larger than the noise. At the lowest loss the roll rates support (4.8%), approving everyone earns NT$248 per account a month and the break-even probability is 27.6%, so 79% of accounts are worth approving; at the base case (11.8%) approving everyone loses NT$473 and 45% are worth approving; at the 15% stress loss only 28% are.

## The policy rate and the cut-off

| Policy rate on | Rate | Margin a month | Break-even probability | Approved | Default rate of approved | Profit per account a month | If the Oct 2021 cut-off were kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 October 2021 | 7.25% | 2.21% | 15.9% | 56% | 10.2% | NT$263 | NT$263 |
| 8 July 2022 | 15% | 1.57% | 11.8% | 36% | 8.3% | NT$74 | NT$51 |
| 27 June 2023 | 22% | 0.99% | 7.7% | 7% | 4.0% | NT$23 | -NT$141 |
| 17 December 2024 | 13% | 1.74% | 12.9% | 42% | 9.1% | NT$122 | NT$106 |
| 16 December 2025 | 10.5% | 1.94% | 14.2% | 49% | 9.5% | NT$183 | NT$174 |
| 27 October 2026 | 12% | 1.82% | 13.4% | 45% | 9.2% | NT$154 | NT$133 |

At 22% the margin on a card balance is thin, so only the safest accounts are worth holding (7% of the book); at 7.25% 56% are. At the peak, keeping the earlier cut-off would have lost money on the accounts it approved (-NT$141 per account a month, against NT$23 with the cut-off reset). The reset cut-off earned more than the Oct 2021 one on 5 of the 5 later dates.

## Beyond the scorecard

**Explainable machine learning.** A boosting model told which way 19 features must push the probability reaches an AUC of 0.792: level with free boosting's (95% range of the gap -0.0015 to +0.0016; within the noise) and 0.023 above the scorecard's (+0.0173 to +0.0304; a real gap). At the break-even cut-off it earns NT$185 per account a month: NT$31 more than the scorecard's NT$154 (95% range of the gap -NT$27 to NT$93; a difference that could be chance on a test this size), against NT$191 for free boosting. Moving each constrained feature across its range for 300 sampled accounts (5,700 account-and-feature checks), free boosting moved the wrong way somewhere along the range in 4,191 of them (74%) and the monotone model in none; where free boosting did, the probability fell below an earlier level by 0.6 percentage points on average and by as much as 9.2 percentage points.

**Stress tests and pricing.** With defaults 50% higher and the cut-off kept, expected profit goes from NT$161 to -NT$25 a month; re-setting the cut-off gives NT$35 on 17% of accounts. At the 2023 peak policy rate the kept cut-off gives -NT$85 and a re-set one NT$5; with defaults 50% higher as well, -NT$260 against NT$0. At the current card APR of 44%, approving pays for the safest 4 of 10 tenths of accounts. The other 6 would need APRs from 44% to 394% to break even. The riskiest need prices far above the one card rate checked here (MCB's published 44%), so for them the choice is to decline or to limit the exposure, not to price for risk.

**Where the models differ.** The scorecard and gradient boosting decide alike on 86% of accounts. Of the 13.7% they split on, those only gradient boosting approves defaulted at 9.6% and carry an average balance of NT$34,703, against 17.3% and NT$113,995 for those only the scorecard approves; approving the first group would have earned NT$8 per account of the portfolio, the second -NT$29.

**Another lender's accounts.** A lender whose accounts miss a payment 8.0% of the time, instead of 22.1%, would put the cut-off at 467 points instead of 535 and approve 82% of accounts instead of 45%, with the same economics and the same ranking.

## What could be wrong

- **The loss is still the weakest number, but it now rests on the file's own roll rates.** "Default" in the data means a missed payment, not a write-off, so the loss comes from roll rates over April to August, the start of a crisis, and later write-offs were probably higher: read it as a floor (4.8% to 11.8% across the three write-off lines, 11.8% used, 15% kept as a stress). Across the grid in the [README](../README.md) (APR 30% to 44%, loss 4.8% to 30%) the break-even probability runs from 2.6% to 27.6%, and the share worth approving from 0% to 79%.
- **The accounts are not Pakistani and are from a crisis-era book.** A lender would refit the scorecard on its own accounts and test it on a later period. The data have only one.
- **Fairness.** At the break-even cut-off, 3 of 11 groups are approved at under four-fifths of the best-treated group's rate: age: 21-29, age: 50 and over and education: High school; for education: High school the whole 95% bootstrap range of the ratio is under 0.80, for the rest it reaches above it. Women default less than men in this file (20.8% against 24.2%) and are approved more often (47% against 41%); the model is calibrated for both (predicted ÷ observed 1.00 for women, 0.98 for men). The flagged groups have higher default rates in this file than the best-treated group of the same attribute, and the model is calibrated for them (predicted ÷ observed between 0.93 and 1.02), so the gaps in approval mostly follow gaps in observed default rates, not a model that is harsher than the data on those groups. The largest calibration gap among the groups judged is 7% (age: 50 and over). None of this makes the gaps acceptable: they are the reason to review features and cut-off by group at every re-set.
- **Selection.** These are accounts the bank had approved, so the model has never seen applicants it would have rejected.

## Next steps for a real lender

1. Refit the scorecard on the lender's own accounts and test it on a later period, not a random hold-out.
2. Replace the roll-rate loss with the lender's own roll rates, write-offs and recoveries (here the status codes could not be tied to months and the history was six months).
3. Run the new cut-off alongside the old one on a small share of accounts before switching.
4. Monitor the score distribution monthly (population stability index above 0.10 is a warning, above 0.25 a reason to refit), and the default rate of each score band against its prediction.
5. Review the audit by group at each re-set of the cut-off, and keep the reason codes with every decline.
