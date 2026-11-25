"""The second layer: ways of testing and using the scorecard that go beyond ranking accounts.

  * monotone_violations: does a boosting model told "more of this never means less risk" really obey?
  * swap_sets: of two models, which accounts does each approve that the other would decline, and who was right?
  * stress_table: what happens to profit if defaults rise, or money gets dearer, or both, with the cut-off kept and with it re-set?
  * pricing_table: the card APR each group of accounts would need for approving it to break even.
  * local_rate_table: how the cut-off moves when the scorecard is re-anchored to a lender whose accounts miss payments at another rate.

The stress, pricing and re-anchoring tables are worked out from the model's own probabilities, so they are expected values, not
outcomes: the file has no accounts that lived through a different year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .economics import Economics, policy_rate_on

EPS = 1e-12


# ---------------------------------------------------------------------------
# Monotone boosting
# ---------------------------------------------------------------------------
def monotone_violations(model, frame: pd.DataFrame, monotone: dict, accounts: int = 300, grid_points: int = 15, seed: int = config.SEED) -> pd.DataFrame:
    """For each constrained feature the model uses: a sample of accounts, the feature moved across its range with everything else held
    fixed, the number of accounts whose predicted probability moves the wrong way anywhere along the way, and how far it moves: the
    fall of the probability below an earlier level, as a share (0.01 is a percentage point), the largest and the average among the
    accounts that broke the rule."""
    sample = frame.sample(min(accounts, len(frame)), random_state=seed).reset_index(drop=True)
    rows = []
    for feature, direction in monotone.items():
        if feature not in model.columns:
            continue
        grid = np.unique(np.quantile(frame[feature], np.linspace(0, 1, grid_points)))
        if len(grid) < 2:
            continue
        stacked = pd.concat([sample.assign(**{feature: value}) for value in grid], ignore_index=True)
        p = model.predict_proba(stacked).reshape(len(grid), len(sample))                 # one row per grid value, one column per account
        wrong = (direction * np.diff(p, axis=0) < -EPS).any(axis=0)
        q = direction * p                                                                # should never fall as the feature rises
        fall = (np.maximum.accumulate(q, axis=0) - q).max(axis=0)                        # the deepest drop below an earlier level, per account
        rows.append({"feature": feature, "direction": direction, "accounts": len(sample), "violations": int(wrong.sum()),
                     "largest_fall": float(fall[wrong].max()) if wrong.any() else 0.0, "average_fall": float(fall[wrong].mean()) if wrong.any() else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Swap sets
# ---------------------------------------------------------------------------
def swap_sets(y, balance, approve_a, approve_b, econ: Economics, names: tuple[str, str]) -> pd.DataFrame:
    """The accounts both models approve, only the first, only the second and neither: how many, how many of them defaulted, and
    what they earned (realised, per account of the whole portfolio) under `econ`."""
    y, balance = np.asarray(y), np.asarray(balance, dtype=float)
    a, b = np.asarray(approve_a, dtype=bool), np.asarray(approve_b, dtype=bool)
    realised = balance * np.where(y == 1, -econ.loss, econ.margin)
    cells = (("Both approve", a & b), (f"Only {names[0]} approves", a & ~b), (f"Only {names[1]} approves", ~a & b), ("Neither approves", ~a & ~b))
    rows = []
    for label, mask in cells:
        rows.append({"cell": label, "accounts": int(mask.sum()), "share": float(mask.mean()), "default_rate": float(y[mask].mean()) if mask.any() else np.nan,
                     "mean_balance": float(balance[mask].mean()) if mask.any() else np.nan, "profit_if_approved": float(realised[mask].sum() / len(y))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------
def shift_odds(probability, factor: float) -> np.ndarray:
    """Probabilities with their odds multiplied by `factor` (for small probabilities that is nearly the same as multiplying them)."""
    odds = np.asarray(probability, dtype=float) / (1 - np.asarray(probability, dtype=float)) * factor
    return odds / (1 + odds)


def expected_profit(probability, balance, econ: Economics, approved) -> float:
    """Expected profit per account of the whole portfolio from approving the accounts in `approved`."""
    p, e = np.asarray(probability, dtype=float), np.asarray(balance, dtype=float)
    gain = e * ((1 - p) * econ.margin - p * econ.loss)
    return float(gain[np.asarray(approved, dtype=bool)].sum() / len(p))


def stress_table(probability, balance, base: Economics, scenarios, policy: pd.DataFrame) -> pd.DataFrame:
    """For each scenario (a name, a factor on the odds of default, and the policy-rate date that sets the funding cost, or None for the
    current rate): the expected profit when the lender keeps its present cut-off, and when it re-sets the cut-off to the scenario."""
    p0 = np.asarray(probability, dtype=float)
    kept = p0 <= base.break_even                                   # the decisions made before the shock
    rows = []
    for name, factor, date in scenarios:
        econ = base if date is None else base.with_(funding=policy_rate_on(date, policy))
        p = shift_odds(p0, factor)
        reset = p <= econ.break_even
        rows.append({"scenario": name, "odds_factor": factor, "policy_rate": econ.funding, "margin_month": econ.margin, "break_even_pd": econ.break_even,
                     "mean_probability": float(p.mean()), "approved_kept": float(kept.mean()),
                     "default_rate_kept": float(p[kept].mean()) if kept.any() else np.nan, "profit_kept": expected_profit(p, balance, econ, kept),
                     "approved_reset": float(reset.mean()), "default_rate_reset": float(p[reset].mean()) if reset.any() else np.nan,
                     "profit_reset": expected_profit(p, balance, econ, reset)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Risk-based pricing
# ---------------------------------------------------------------------------
def pricing_table(probability, balance, econ: Economics, groups: int = config.PRICING_GROUPS) -> pd.DataFrame:
    """The accounts cut into equal groups from the safest, and for each the card APR that makes approving it break even:
    the margin on a good account must cover p x loss / (1 - p), and the APR must also cover funding, running cost and the share
    of balances that never revolves."""
    p, e = np.asarray(probability, dtype=float), np.asarray(balance, dtype=float)
    order = pd.Series(p).rank(method="first")
    group = pd.qcut(order, groups, labels=False)
    rows = []
    for g in range(groups):
        mask = (group == g).to_numpy()
        mean_p = float(p[mask].mean())
        margin_needed = mean_p * econ.loss / (1 - mean_p)
        required_apr = (12 * margin_needed + econ.funding + econ.opex) / econ.revolve_share
        rows.append({"group": g + 1, "accounts": int(mask.sum()), "mean_probability": mean_p, "mean_balance": float(e[mask].mean()),
                     "required_margin_month": margin_needed, "required_apr": required_apr, "pays_at_current_apr": bool(required_apr <= econ.apr),
                     "expected_profit_per_account": float((e[mask] * ((1 - p[mask]) * econ.margin - p[mask] * econ.loss)).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Re-anchoring to another lender's default rate
# ---------------------------------------------------------------------------
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def local_rate_table(probability, balance, econ: Economics, scaling, rates, from_rate: float) -> pd.DataFrame:
    """The scorecard was fitted on accounts that miss a payment at `from_rate`. A lender whose accounts miss at another rate can keep the
    ranking and move every log-odds by the difference between the two base log-odds (the usual correction for a change in the base
    rate). For each rate: the break-even cut-off in the original probabilities and in score points, the share it approves, and the
    expected profit and default rate of the approved accounts at the new rate."""
    p0, e = np.asarray(probability, dtype=float), np.asarray(balance, dtype=float)
    rows = []
    for rate in [*rates, from_rate]:
        delta = float(logit(rate) - logit(from_rate))
        p = expit(logit(p0) + delta)
        approved = p <= econ.break_even
        cutoff_original = float(expit(logit(econ.break_even) - delta))
        rows.append({"local_default_rate": rate, "log_odds_shift": delta, "break_even_pd": econ.break_even, "cutoff_probability_in_original_terms": cutoff_original,
                     "cutoff_score": int(np.ceil(float(scaling.score(cutoff_original)))), "approved_share": float(approved.mean()),
                     "expected_default_rate_approved": float(p[approved].mean()) if approved.any() else np.nan, "mean_probability": float(p.mean()),
                     "expected_profit_per_account": expected_profit(p, e, econ, approved)})
    return pd.DataFrame(rows)
