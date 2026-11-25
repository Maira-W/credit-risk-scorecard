"""Roll rates and the loss on a missed payment, on chains whose answers are known."""

import numpy as np
import pandas as pd
import pytest

from creditrisk import config, rollrates

# Moves from one month to the next, by state (rows: from, columns: to), as counts. The first row is "current", the rest are the late statuses.
LADDER = np.array([[950, 50, 0, 0, 0, 0], [300, 600, 100, 0, 0, 0], [150, 350, 200, 300, 0, 0],
                   [50, 250, 100, 300, 300, 0], [30, 200, 70, 100, 150, 450], [10, 250, 10, 10, 10, 710]])
TRUE = rollrates.matrix(LADDER)


def test_the_status_ladder():
    status = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert list(rollrates.state_of(status)) == [0, 0, 0, 1, 1, 2, 3, 4, 5, 5, 5]


def test_counts_by_hand():
    paths = np.array([[0, 0, 1, 1, 2],
                      [1, 0, 0, 0, 0]])
    counts = rollrates.counts_from(paths)
    assert counts.sum() == 8 and counts[0, 0] == 4 and counts[0, 1] == 1 and counts[1, 1] == 1 and counts[1, 2] == 1 and counts[1, 0] == 1
    assert counts[2, :].sum() == 0


def test_rows_become_probabilities_and_a_state_never_seen_stays_where_it_is():
    counts = np.zeros((6, 6), dtype=int)
    counts[0] = [8, 2, 0, 0, 0, 0]
    counts[1] = [3, 6, 1, 0, 0, 0]
    p = rollrates.matrix(counts)
    assert p[0, 1] == pytest.approx(0.2) and p[1].sum() == pytest.approx(1) and p[3, 3] == 1.0 and p[5, 5] == 1.0


def chain(q_worse=0.2, stay=0.5, cure=0.3) -> np.ndarray:
    """From the first late state: stay, cure, or move to the second late state; every other state stays where it is."""
    p = np.eye(6)
    p[1] = [cure, stay, q_worse, 0, 0, 0]
    return p


@pytest.mark.parametrize("months", [1, 3, 12])
def test_reaching_a_worse_status_by_hand(months):
    # each month the account is still in the first late state it moves on with 0.2 and stays with 0.5: 0.2 x (1 - 0.5^months) / 0.5
    expected = 0.2 * (1 - 0.5 ** months) / 0.5
    assert rollrates.reach_probability(chain(), 1, 3, months) == pytest.approx(expected)


def test_the_chance_grows_with_the_months_and_falls_with_the_write_off_line():
    by_months = [rollrates.reach_probability(TRUE, 1, 5, h) for h in (1, 6, 12, 24)]
    assert by_months == sorted(by_months) and by_months[0] == 0.0          # status 5 cannot be reached from status 2 in one step
    by_line = [rollrates.reach_probability(TRUE, 1, s, 12) for s in (3, 4, 5, 6)]
    assert by_line == sorted(by_line, reverse=True)


def test_starting_from_a_mixture_is_the_average_of_the_starting_points():
    mix = np.array([0.0, 0.7, 0.3, 0, 0, 0])
    expected = 0.7 * rollrates.reach_probability(TRUE, 1, 4, 12) + 0.3 * rollrates.reach_probability(TRUE, 2, 4, 12)
    assert rollrates.reach_probability(TRUE, mix, 4, 12) == pytest.approx(expected)


def test_a_status_outside_the_ladder_is_an_error():
    with pytest.raises(ValueError):
        rollrates.reach_probability(chain(), 1, 1, 6)
    with pytest.raises(ValueError):
        rollrates.reach_probability(chain(), 1, 7, 6)


def test_the_loss_is_the_share_never_recovered_times_the_chance_plus_the_cost():
    assert rollrates.loss_given_missed(0.1, lgd=0.8, collection_cost=0.02) == pytest.approx(0.1)
    assert rollrates.loss_given_missed(0.0) == config.ECON_COLLECTION_COST


def test_a_missed_payment_lands_in_a_late_state_by_hand():
    p = np.eye(6)
    p[0] = [0.9, 0.1, 0, 0, 0, 0]
    p[1] = [0.3, 0.5, 0.2, 0, 0, 0]
    p[2] = [0.2, 0.2, 0.2, 0.4, 0, 0]
    # a current account lands on the first late level; a late one stays or gets worse, with the cure (which a missed payment rules out) set aside
    assert rollrates.october_entry(p, np.array([0])) == pytest.approx([0, 1, 0, 0, 0, 0])
    assert rollrates.october_entry(p, np.array([1])) == pytest.approx([0, 0.5 / 0.7, 0.2 / 0.7, 0, 0, 0])
    assert rollrates.october_entry(p, np.array([2])) == pytest.approx([0, 0.2 / 0.8, 0.2 / 0.8, 0.4 / 0.8, 0, 0])
    mixed = rollrates.october_entry(p, np.array([0, 0, 1, 2]))                        # half were current, a quarter at each of the next two states
    assert mixed == pytest.approx(0.5 * np.array([0, 1, 0, 0, 0, 0]) + 0.25 * np.array([0, 5 / 7, 2 / 7, 0, 0, 0]) + 0.25 * np.array([0, 0.25, 0.25, 0.5, 0, 0]))
    assert mixed.sum() == pytest.approx(1) and mixed[0] == 0


def test_a_state_that_never_left_for_a_late_state_lands_where_it_is_or_on_the_first_late_level():
    p = np.eye(6)
    p[0] = [1, 0, 0, 0, 0, 0]
    p[2] = [1, 0, 0, 0, 0, 0]
    assert rollrates.october_entry(p, np.array([0])) == pytest.approx([0, 1, 0, 0, 0, 0])
    assert rollrates.october_entry(p, np.array([2])) == pytest.approx([0, 0, 1, 0, 0, 0])
    assert rollrates.october_entry(p, np.array([], dtype=int)).sum() == 0                  # no defaulters: nothing to start


def test_starting_where_the_label_says_gives_more_than_starting_from_september():
    """Defaulters that were current in September have missed a payment, so they start late: the chance of a write-off is higher than if they
    were left at 'current' (where it takes a first miss to start the slide)."""
    september = np.array([0] * 5 + [1] * 5)
    from_september = rollrates.reach_probability(TRUE, rollrates.entry_distribution(september), 4, 12)
    from_october = rollrates.reach_probability(TRUE, rollrates.october_entry(TRUE, september), 4, 12)
    assert from_october > 1.5 * from_september


def simulate_states(n: int, transitions: np.ndarray, months: int, seed: int = 0) -> np.ndarray:
    """The state of n accounts in each of `months` months, every account following the chain."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n, months), dtype=int)
    states[:, 0] = rng.choice(6, n, p=[0.8, 0.12, 0.04, 0.02, 0.01, 0.01])
    cumulative = np.cumsum(transitions, axis=1)
    for t in range(1, months):
        u = rng.random(n)
        states[:, t] = (u[:, None] > cumulative[states[:, t - 1]]).sum(axis=1).clip(max=5)
    return states


def as_frame(states: np.ndarray) -> pd.DataFrame:
    """The first six months as repayment statuses (state k from 1 up is status k + 1): April to August, then September."""
    status = np.where(states[:, :6] == 0, 0, states[:, :6] + 1)
    frame = pd.DataFrame({f"status_{m}": status[:, i] for i, m in enumerate(rollrates.MONTHS)})
    frame["status_sep"] = status[:, 5]
    return frame


def test_the_estimate_matches_what_really_happens_to_accounts_that_miss_a_payment():
    """Accounts follow a known chain for 19 months. The first six are the data, October (month 7) decides who 'defaulted', and the twelve
    months after it are the truth that the estimate, which sees only the six, has to match."""
    states = simulate_states(60_000, TRUE, 19)
    frame, default = as_frame(states), states[:, 6] >= 1
    table, details = rollrates.evidence(frame, default, statuses=(4, 5), months=12, draws=100, seed=3)
    assert np.allclose(details["matrix"], TRUE, atol=0.05)                    # a late state's row rests on a few thousand moves
    for row in table.itertuples():
        truth = float((states[default, 6:19] >= row.write_off_status - 1).any(axis=1).mean())      # reached the line in October or in the twelve months after
        assert abs(row.probability - truth) < 0.015, (row.write_off_status, row.probability, truth)
        assert row.low - 0.01 <= truth <= row.high + 0.01


def test_evidence_is_reproducible_and_laid_out_as_documented():
    states = simulate_states(4000, TRUE, 7)
    frame, default = as_frame(states), states[:, 6] >= 1
    a, _ = rollrates.evidence(frame, default, draws=30, seed=5)
    b, _ = rollrates.evidence(frame, default, draws=30, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["write_off_status"]) == list(config.WRITEOFF_CHECKS) and (a["months"] == config.ECON_HORIZON).all()
    assert ((a["low"] <= a["probability"]) & (a["probability"] <= a["high"])).all() and a["probability"].is_monotonic_decreasing
    assert np.allclose(a["loss"], config.ECON_LGD * a["probability"] + config.ECON_COLLECTION_COST)
    assert ((a["loss_low"] <= a["loss"]) & (a["loss"] <= a["loss_high"])).all()


def test_the_split_by_september_state_adds_up_to_the_whole():
    states = simulate_states(8000, TRUE, 7)
    frame, default = as_frame(states), states[:, 6] >= 1
    table, details = rollrates.evidence(frame, default, draws=20, seed=1)
    split = details["split"]
    assert list(split.columns) == ["group", "share", "write_off_status", "months", "probability", "low", "high", "loss", "loss_low", "loss_high"]
    assert len(split) == 2 * len(config.WRITEOFF_CHECKS) and set(split["group"]) == set(rollrates.GROUPS[1:])
    shares = split.drop_duplicates("group").set_index("group")["share"]
    assert shares["Current in September"] == pytest.approx((states[default, 5] == 0).mean()) and shares.sum() == pytest.approx(1)
    current, late = (split[split["group"] == g].set_index("write_off_status") for g in rollrates.GROUPS[1:])
    for row in table.itertuples():            # the chance is linear in where the accounts start, so the whole is the share-weighted average of the two parts
        whole = shares["Current in September"] * current.loc[row.write_off_status, "probability"] + shares["Late in September"] * late.loc[row.write_off_status, "probability"]
        assert row.probability == pytest.approx(whole)
    assert (late["probability"] > current["probability"]).all()               # already being late is worse than a first miss


def test_tables_have_the_documented_shape():
    counts = np.arange(36).reshape(6, 6) + 1
    long = rollrates.matrix_table(counts)
    assert len(long) == 36 and list(long.columns) == ["from_state", "to_state", "moves", "probability"]
    assert long.groupby("from_state")["probability"].sum().round(9).eq(1).all() and long["moves"].sum() == counts.sum()
    horizons = rollrates.horizon_table(rollrates.matrix(counts), np.array([0, 1, 0, 0, 0, 0]))
    assert len(horizons) == 12
    for _, part in horizons.groupby("write_off_status"):
        assert part.sort_values("months")["probability"].is_monotonic_increasing
