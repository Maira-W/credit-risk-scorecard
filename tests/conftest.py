"""Shared test data: made-up accounts with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk import config


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A made-up table in the shape `build.load_raw` returns (readable column names), valid for `build.validate`."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 400_000], n)
    status = rng.choice([-2, -1, 0, 1, 2, 3], size=(n, 6), p=[0.1, 0.2, 0.45, 0.1, 0.1, 0.05])
    bills = (limit[:, None] * rng.uniform(-0.05, 1.1, (n, 6))).round().astype(int)
    paid = (np.clip(bills, 0, None) * rng.uniform(0, 1, (n, 6))).round().astype(int)
    risk = 0.12 + 0.12 * (status[:, 0] >= 1) + 0.15 * (status[:, 0] >= 2)
    frame = pd.DataFrame({"id": np.arange(1, n + 1), "limit": limit, "sex_code": rng.choice([1, 2], n), "education_code": rng.choice([0, 1, 2, 3, 4, 5], n),
                          "marriage_code": rng.choice([0, 1, 2, 3], n), "age": rng.integers(21, 70, n)})
    for i, m in enumerate(config.MONTHS):
        frame[f"status_{m}"] = status[:, i]
        frame[f"bill_{m}"] = bills[:, i]
        frame[f"paid_{m}"] = paid[:, i]
    frame["default"] = (rng.uniform(size=n) < risk).astype(int)
    return frame


@pytest.fixture(scope="session")
def raw_small() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def logistic_world():
    """20,000 accounts whose default really follows a logistic model of five features: two that matter, two that do not,
    and one that is a noisy copy of the first. Returns the table, the outcome and the true probabilities."""
    rng = np.random.default_rng(7)
    n = 20_000
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.normal(size=n)
    x4 = x1 + rng.normal(scale=0.3, size=n)                         # a noisy copy of x1
    x5 = rng.integers(0, 4, n)
    logit = -1.4 + 0.9 * x1 + 0.18 * (x2 - 5)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}), y, p
