"""Roll rates: how often an account that misses a payment recovers, stays late or gets worse, and the loss that implies.

The repayment status of the five months April to August, which are coded on one basis (a status of 1 hardly appears in them; see
build.py), is read as a ladder of states: current (a status of 0 or below), then the late statuses 2, 3, 4, 5 and 6-or-more.
Counting how accounts move from one month to the next gives a transition matrix. Treating that matrix as steady, the chance that a
missed payment grows into a write-off within a year follows from raising it to a power, and the loss on a missed payment is that chance
times the share of the balance never recovered, plus a cost of collection. That is where `economics` gets its loss, in place of a guess.

Where the chain starts matters. The label is a missed payment in October, so every account that "defaulted" is late in October, whatever
it was in September: a current account has just slipped to the first late level, a late one has stayed or got worse and did not cure.
Starting the chain from September's state would treat the many defaulters that were current in September as not having missed a payment
yet, and would let the late ones cure in the step the label says they did not. Each defaulter is therefore moved one month on to where it
landed in October (the matrix's own odds of each late state from where it stood, with the odds of being current set aside), and the
chain is followed from there for a year (`october_entry`).

What it rests on, all stated in config: where the write-off line sits (status 4 in the base case, the conservative end), how much of a
written-off balance is never recovered, what collection costs, and that the roll rates of April to August hold for the next year. The
file is the start of a card-debt crisis, so later roll rates were probably worse; the loss is better read as a floor than as a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MONTHS = ("apr", "may", "jun", "jul", "aug")
N_STATES = 6
NAMES = ["Current", "Late, status 2", "Late, status 3", "Late, status 4", "Late, status 5", "Late, status 6 or more"]
GROUPS = ("All missed payments", "Current in September", "Late in September")


def state_of(status) -> np.ndarray:
    """0 for current (a status of 0 or below), 1 to 5 for the late statuses 2, 3, 4, 5 and 6 or more (a status of 1 joins 2)."""
    s = np.asarray(status)
    return np.where(s <= 0, 0, np.clip(s, 2, 6) - 1).astype(int)


def paths(frame: pd.DataFrame) -> np.ndarray:
    """One row per account: its state in each of the five months, oldest first."""
    return np.column_stack([state_of(frame[f"status_{m}"]) for m in MONTHS])


def counts_from(account_paths: np.ndarray) -> np.ndarray:
    """How many times an account was in state i one month and in state j the next, over every pair of months and every account."""
    before, after = account_paths[:, :-1].ravel(), account_paths[:, 1:].ravel()
    return np.bincount(before * N_STATES + after, minlength=N_STATES * N_STATES).reshape(N_STATES, N_STATES)


def matrix(counts) -> np.ndarray:
    """Counts turned into probabilities, row by row. A state that was never seen stays where it is."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    probabilities[totals[:, 0] == 0] = np.eye(N_STATES)[totals[:, 0] == 0]
    return probabilities


def reach_probability(transitions: np.ndarray, start, status: int, months: int) -> float:
    """The chance of reaching `status` (or any worse one) at some point within `months`, from a state (an int) or from a distribution
    over states. Once an account gets there it is counted and stays there."""
    target = status - 1
    if not 1 <= target < N_STATES:
        raise ValueError("status must be between 2 and 6")
    q = transitions.copy()
    q[target:, :] = 0
    q[target:, target:] = np.eye(N_STATES - target)
    v = np.zeros(N_STATES)
    if np.ndim(start) == 0:
        v[int(start)] = 1.0
    else:
        v = np.asarray(start, dtype=float)
    return float((v @ np.linalg.matrix_power(q, months))[target:].sum())


def entry_distribution(states: np.ndarray) -> np.ndarray:
    """The share of the given accounts in each state."""
    return np.bincount(states, minlength=N_STATES) / max(len(states), 1)


def october_entry(transitions: np.ndarray, september_states: np.ndarray) -> np.ndarray:
    """Where accounts that missed their October payment stood in October, given where each stood in September.

    Having missed a payment, an account is late in October. Its landing state follows the matrix's odds of each late state from where it
    stood in September, with the odds of being current set aside and the rest scaled up to one: a current account lands on the first late
    level, a late one stays or gets worse. A state the matrix never saw leave for a late state stays where it is (or goes to the first
    late level if it was current)."""
    weights = entry_distribution(np.asarray(september_states))
    late = np.array(transitions, dtype=float)
    late[:, 0] = 0.0
    totals = late.sum(axis=1, keepdims=True)
    landing = np.divide(late, totals, out=np.zeros_like(late), where=totals > 0)
    unseen = totals[:, 0] == 0
    landing[unseen] = np.eye(N_STATES)[np.maximum(np.arange(N_STATES), 1)][unseen]
    return weights @ landing


def loss_given_missed(probability: float, lgd: float = config.ECON_LGD, collection_cost: float = config.ECON_COLLECTION_COST) -> float:
    """Loss on an account that misses a payment, as a share of its balance: the chance it is written off times the share never
    recovered, plus the cost of collecting and of the income lost while it is late."""
    return lgd * probability + collection_cost


def evidence(frame: pd.DataFrame, default, statuses=config.WRITEOFF_CHECKS, months: int = config.ECON_HORIZON, lgd: float = config.ECON_LGD,
             collection_cost: float = config.ECON_COLLECTION_COST, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED) -> tuple[pd.DataFrame, dict]:
    """For each write-off line, the chance that an account that misses a payment is written off within `months` and the loss that
    implies, with a 95% bootstrap range (accounts are resampled, so the uncertainty of both the matrix and the starting point counts).

    The accounts that really missed their payment are started where they landed in October (`october_entry`). Returns the table for
    all of them, and a dictionary with the matrix, the headline rates and `split`: the same table for the defaulters that were current in
    September (their first missed payment) and for those that were already late."""
    account_paths = paths(frame)
    september = state_of(frame["status_sep"])
    defaulted = np.asarray(default).astype(bool)
    n = len(frame)

    def estimate(pick: np.ndarray) -> np.ndarray:
        """The chance of reaching each write-off line, for all missed payments, for those current in September and for those late: groups x lines."""
        p = matrix(counts_from(account_paths[pick]))
        stood = september[pick][defaulted[pick]]
        chosen = (stood, stood[stood == 0], stood[stood > 0])
        return np.array([[reach_probability(p, october_entry(p, group), s, months) for s in statuses] for group in chosen])

    point = estimate(np.arange(n))
    rng = np.random.default_rng(seed)
    resampled = np.array([estimate(rng.integers(0, n, n)) for _ in range(draws)])
    low, high = np.percentile(resampled, 2.5, axis=0), np.percentile(resampled, 97.5, axis=0)

    def table_for(g: int) -> pd.DataFrame:
        return pd.DataFrame({"write_off_status": list(statuses), "months": months, "probability": point[g], "low": low[g], "high": high[g],
                             "loss": [loss_given_missed(p, lgd, collection_cost) for p in point[g]],
                             "loss_low": [loss_given_missed(p, lgd, collection_cost) for p in low[g]],
                             "loss_high": [loss_given_missed(p, lgd, collection_cost) for p in high[g]]})

    table = table_for(0)
    stood = september[defaulted]
    shares = (float((stood == 0).mean()) if len(stood) else 0.0, float((stood > 0).mean()) if len(stood) else 0.0)
    split = pd.concat([table_for(g).assign(group=GROUPS[g], share=shares[g - 1]) for g in (1, 2)], ignore_index=True)[["group", "share", *table.columns]]
    counts = counts_from(account_paths)
    p = matrix(counts)
    details = {"counts": counts, "matrix": p, "entry": entry_distribution(stood), "entry_october": october_entry(p, stood), "split": split,
               "lgd": lgd, "collection_cost": collection_cost, "months": months,
               "p_current_to_late": float(1 - p[0, 0]), "p_cure_from_first_late": float(p[1, 0]), "p_stay_first_late": float(p[1, 1]),
               "p_worse_from_first_late": float(p[1, 2:].sum())}
    return table, details


def horizon_table(transitions: np.ndarray, entry: np.ndarray, statuses=(3, 4, 5, 6), horizons=(6, 12, 24)) -> pd.DataFrame:
    """The chance of reaching each late status within 6, 12 and 24 months of the missed payment, from where those accounts landed
    (`october_entry`)."""
    rows = [{"write_off_status": s, "months": h, "probability": reach_probability(transitions, entry, s, h)} for s in statuses for h in horizons]
    return pd.DataFrame(rows)


def matrix_table(counts: np.ndarray) -> pd.DataFrame:
    """The transition matrix in long form: from, to, number of moves and the probability."""
    p = matrix(counts)
    return pd.DataFrame([{"from_state": NAMES[i], "to_state": NAMES[j], "moves": int(counts[i, j]), "probability": float(p[i, j])}
                         for i in range(N_STATES) for j in range(N_STATES)])
