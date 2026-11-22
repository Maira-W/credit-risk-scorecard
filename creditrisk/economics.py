"""What a cut-off is worth: the profit of approving an account, and where approving stops paying.

Per unit of balance, over one billing month, an account that pays earns a margin and an account that defaults loses a share of its
balance:   profit = margin if it pays,  -loss if it defaults.
    margin = (yield on balances - funding cost - running cost) / 12,   yield = card APR x the share of balances that revolve.
An account with probability of default p is worth (1 - p) x margin - p x loss per unit of balance, which is positive only while
    p < margin / (margin + loss),   the break-even probability.
Approving everyone whose probability is at or below a cut-off therefore pays until the cut-off reaches the break-even probability.
The funding cost is the policy rate: money lent on a card is money not put in government paper, which pays about the policy rate.
Balances are the September statement (never below zero). All the rates are assumptions except where config names a source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Economics:
    apr: float = config.ECON_APR
    revolve_share: float = 1.0       # the share of balances that earn the APR: the rest is repaid within the interest-free period
    funding: float = 0.12            # a year, as a share of balances
    opex: float = config.ECON_OPEX
    loss: float = config.ECON_LOSS

    @property
    def yield_(self) -> float:
        return self.apr * self.revolve_share

    @property
    def margin(self) -> float:
        """Net monthly earnings on a unit of balance of an account that pays."""
        return (self.yield_ - self.funding - self.opex) / 12

    @property
    def break_even(self) -> float:
        """The probability of default above which approving an account loses money on average (0 if even a perfect account does)."""
        return self.margin / (self.margin + self.loss) if self.margin > 0 else 0.0

    def with_(self, **changes) -> Economics:
        return replace(self, **changes)


def revolve_share(df: pd.DataFrame) -> float:
    """The share of statement balances still unpaid after the next month's payment, pooled over the five month pairs in the file.
    (A payment in a month settles the statement of the month before.)"""
    owed = paid = 0.0
    for month, before in zip(config.MONTHS[:-1], config.MONTHS[1:], strict=True):
        due = df[f"bill_{before}"].clip(lower=0)
        owed += due.sum()
        paid += np.minimum(df[f"paid_{month}"], due).sum()
    return float(1 - paid / owed)


def exposure(df: pd.DataFrame) -> np.ndarray:
    """The balance at risk: the September statement, or zero for an account in credit."""
    return df["bill_sep"].clip(lower=0).to_numpy(dtype=float)


def policy_rate_on(date: str, table: pd.DataFrame | None = None) -> float:
    """The policy rate (as a share, 0.12 for 12%) in force on a date: the latest change on or before it."""
    table = table if table is not None else pd.read_csv(config.POLICY_RATE_FILE, parse_dates=["effective_date"])
    row = table[table["effective_date"] <= pd.Timestamp(date)].sort_values("effective_date").iloc[-1]
    return float(row["policy_rate_pct"]) / 100


# ---------------------------------------------------------------------------
# Profit by cut-off
# ---------------------------------------------------------------------------
def threshold_table(y, probability, balance) -> pd.DataFrame:
    """One row for each distinct probability, from the safest account up: how many accounts have it and how much of their
    balances belong to accounts that paid and to accounts that defaulted. Approving every account at or below a probability
    means taking the rows up to it, so the profit of every possible cut-off follows from this table."""
    y, p, e = np.asarray(y), np.asarray(probability, dtype=float), np.asarray(balance, dtype=float)
    values, inverse = np.unique(p, return_inverse=True)
    size = len(values)
    return pd.DataFrame({
        "probability": values,
        "accounts": np.bincount(inverse, minlength=size),
        "defaults": np.bincount(inverse, weights=y, minlength=size),
        "balance_paid": np.bincount(inverse, weights=e * (1 - y), minlength=size),
        "balance_defaulted": np.bincount(inverse, weights=e * y, minlength=size),
        "balance_x_probability": np.bincount(inverse, weights=e * p, minlength=size),
        "balance": np.bincount(inverse, weights=e, minlength=size),
    })


def coarsen(table: pd.DataFrame, min_share: float = 0.01) -> pd.DataFrame:
    """Merge neighbouring rows so that each holds at least `min_share` of the accounts (the last may hold less). The merged row
    keeps the highest probability in it, which is the cut-off that approves the whole row."""
    minimum = min_share * table["accounts"].sum()
    group, number, running = np.empty(len(table), dtype=int), 0, 0
    for i, accounts in enumerate(table["accounts"].to_numpy()):
        group[i] = number
        running += accounts
        if running >= minimum:                      # this group is big enough: start the next one
            number, running = number + 1, 0
    summed = table.groupby(group).agg({"accounts": "sum", "defaults": "sum", "balance_paid": "sum", "balance_defaulted": "sum",
                                       "balance_x_probability": "sum", "balance": "sum", "probability": "max"})
    return summed[table.columns].reset_index(drop=True)


def profit_curve(table: pd.DataFrame, econ: Economics) -> pd.DataFrame:
    """Profit of approving everyone at or below each cut-off in the table, realised (the outcomes that happened) and expected
    (from the model's own probabilities). Also the share of accounts approved and the default rate among them."""
    c = table[["accounts", "defaults", "balance_paid", "balance_defaulted", "balance_x_probability", "balance"]].cumsum()
    n = table["accounts"].sum()
    return pd.DataFrame({
        "cutoff": table["probability"],
        "approved": c["accounts"],
        "approved_share": c["accounts"] / n,
        "default_rate_approved": c["defaults"] / c["accounts"],
        "realised_profit": econ.margin * c["balance_paid"] - econ.loss * c["balance_defaulted"],
        "expected_profit": econ.margin * (c["balance"] - c["balance_x_probability"]) - econ.loss * c["balance_x_probability"],
        "balance_approved": c["balance"],
    })


def best_cutoff(curve: pd.DataFrame, column: str = "realised_profit") -> pd.Series:
    """The row of the curve with the highest profit, or an empty approval (profit 0) if every cut-off loses money."""
    best = curve.loc[curve[column].idxmax()]
    if best[column] <= 0:
        return pd.Series({"cutoff": -np.inf, "approved": 0, "approved_share": 0.0, "default_rate_approved": np.nan, "realised_profit": 0.0,
                          "expected_profit": 0.0, "balance_approved": 0.0})
    return best


def profit_at(table: pd.DataFrame, econ: Economics, cutoff: float) -> dict:
    """Realised profit, share approved and default rate among the approved when approving every account at or below `cutoff`."""
    chosen = table[table["probability"] <= cutoff]
    n = table["accounts"].sum()
    return {"approved_share": float(chosen["accounts"].sum() / n),
            "default_rate_approved": float(chosen["defaults"].sum() / chosen["accounts"].sum()) if len(chosen) else float("nan"),
            "realised_profit": float(econ.margin * chosen["balance_paid"].sum() - econ.loss * chosen["balance_defaulted"].sum()),
            "balance_approved": float(chosen["balance"].sum())}


def scenarios(table: pd.DataFrame, base: Economics, dates, policy: pd.DataFrame) -> pd.DataFrame:
    """For each policy-rate date: the funding cost, the margin, the break-even probability (which is the cut-off), what approving at
    that cut-off earns on the accounts in `table`, and what the cut-off of the first date would earn under this date's costs."""
    rows = []
    first = None
    n = table["accounts"].sum()
    for date in dates:
        econ = base.with_(funding=policy_rate_on(date, policy))
        first = econ if first is None else first
        mine = profit_at(table, econ, econ.break_even)
        kept = profit_at(table, econ, first.break_even)
        hindsight = best_cutoff(profit_curve(table, econ))
        rows.append({"date": date, "policy_rate": econ.funding, "margin_month": econ.margin, "break_even_pd": econ.break_even,
                     "approved_share": mine["approved_share"], "default_rate_approved": mine["default_rate_approved"],
                     "profit_per_account": mine["realised_profit"] / n,
                     "profit_per_account_first_cutoff_kept": kept["realised_profit"] / n,
                     "approved_share_first_cutoff_kept": kept["approved_share"],
                     "profit_per_account_best_in_hindsight": float(hindsight["realised_profit"]) / n})
    return pd.DataFrame(rows)


def sensitivity(table: pd.DataFrame, base: Economics, losses, aprs) -> pd.DataFrame:
    """The break-even probability, the share approved at it and the profit, for each combination of card APR and loss on a default."""
    rows = []
    n = table["accounts"].sum()
    for apr in aprs:
        for loss in losses:
            econ = base.with_(apr=apr, loss=loss)
            result = profit_at(table, econ, econ.break_even)
            rows.append({"apr": apr, "loss": loss, "margin_month": econ.margin, "break_even_pd": econ.break_even,
                         "approved_share": result["approved_share"], "profit_per_account": result["realised_profit"] / n})
    return pd.DataFrame(rows)
