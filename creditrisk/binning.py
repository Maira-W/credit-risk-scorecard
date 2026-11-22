"""Weight-of-evidence binning: the first step of a credit scorecard.

A feature is cut into a few bins; each bin gets a weight of evidence, WoE = ln(share of all bad accounts in the bin / share of
all good accounts in the bin). A positive WoE means the bin is riskier than the portfolio, a negative one safer. The information
value (IV) of a feature adds up (share of bad - share of good) x WoE over its bins and says how well the bins separate good from
bad: under 0.02 is useless, 0.1 to 0.3 medium, above 0.3 strong.

The bins are found by starting with many small ones and merging neighbours, always the pair that loses least information, until
  1. every bin holds at least `min_share` of the accounts,
  2. the risk rises (or falls) steadily from bin to bin, if the feature is meant to be monotone,
  3. there are at most `max_bins` bins, and
  4. no neighbouring pair could be merged for a loss of less than `iv_tol` of information value.
Categories are first put in order of their default rate, so that only categories with similar risk are merged.
Counts are smoothed by half an account per bin so that a bin with no bad accounts has a finite WoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

SMOOTHING = 0.5
INITIAL_BINS = 30


def woe_iv(bad, good, smoothing: float = SMOOTHING) -> tuple[np.ndarray, np.ndarray]:
    """Weight of evidence and the information-value contribution of each bin, from the number of bad and good accounts in it."""
    bad = np.asarray(bad, dtype=float)
    good = np.asarray(good, dtype=float)
    k = len(bad)
    share_bad = (bad + smoothing) / (bad.sum() + smoothing * k)
    share_good = (good + smoothing) / (good.sum() + smoothing * k)
    woe = np.log(share_bad / share_good)
    return woe, (share_bad - share_good) * woe


def information_value(bad, good) -> float:
    return float(woe_iv(bad, good)[1].sum())


def merge_neighbours(bad, good, *, min_share: float, max_bins: int, iv_tolerance: float, monotone: str | None = None,
                     smoothing: float = SMOOTHING) -> list[list[int]]:
    """Merge neighbouring bins under the four rules in the module description. `bad` and `good` count the starting bins in
    order; the answer lists, for each final bin, the starting bins that went into it. `monotone` is "up" if the default rate
    must not fall from one bin to the next, "down" if it must not rise, or None."""
    bad = [float(v) for v in bad]
    good = [float(v) for v in good]
    total_bad, total_good = sum(bad), sum(good)
    min_count = min_share * (total_bad + total_good)
    groups = [[i] for i in range(len(bad))]

    def part(b: float, g: float, k: int) -> float:
        share_bad = (b + smoothing) / (total_bad + smoothing * k)
        share_good = (g + smoothing) / (total_good + smoothing * k)
        return (share_bad - share_good) * np.log(share_bad / share_good)

    def costs() -> list[float]:
        k = len(groups)
        return [part(bad[i], good[i], k) + part(bad[i + 1], good[i + 1], k) - part(bad[i] + bad[i + 1], good[i] + good[i + 1], k)
                for i in range(k - 1)]

    def merge(i: int) -> None:
        groups[i] = groups[i] + groups.pop(i + 1)
        bad[i] += bad.pop(i + 1)
        good[i] += good.pop(i + 1)

    while len(groups) > 1:
        size = [b + g for b, g in zip(bad, good, strict=True)]
        rate = [b / (b + g) if b + g else 0.0 for b, g in zip(bad, good, strict=True)]
        cost = costs()

        small = [i for i in range(len(groups)) if size[i] < min_count]
        if small:
            i = min(small, key=lambda j: size[j])
            options = ([i - 1] if i > 0 else []) + ([i] if i < len(groups) - 1 else [])
            merge(min(options, key=lambda p: cost[p]))
            continue
        if monotone:
            wrong = [i for i in range(len(groups) - 1) if (rate[i] > rate[i + 1] if monotone == "up" else rate[i] < rate[i + 1])]
            if wrong:
                merge(min(wrong, key=lambda p: cost[p]))
                continue
        cheapest = int(np.argmin(cost))
        if len(groups) > max_bins or cost[cheapest] < iv_tolerance:
            merge(cheapest)
            continue
        break
    return groups


def _format(value: float) -> str:
    return f"{value:.4g}"


def numeric_labels(edges: np.ndarray, integer: bool, low: float | None = None, high: float | None = None) -> list[str]:
    """Readable names for the bins between the cut points. A value equal to a cut point belongs to the bin below it.
    Whole numbers: "2 to 3", "4 or more"; other numbers: "up to 0.2", "over 0.2 up to 0.5", "over 0.5".
    `low` and `high`, the smallest and largest value seen, let the end bins of whole numbers be named by a single value."""
    if len(edges) == 0:
        return ["all values"]
    labels = []
    if integer:
        cuts = [int(np.floor(e)) for e in edges]                      # a cut at 1.5 or at 2.0 sends 1 and 2 to the bin below it
        labels.append(f"{cuts[0]:,}" if low is not None and cuts[0] == low else f"{cuts[0]:,} or less")
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
            labels.append(f"{lo + 1:,}" if hi == lo + 1 else f"{lo + 1:,} to {hi:,}")
        labels.append(f"{cuts[-1] + 1:,}" if high is not None and cuts[-1] + 1 == high else f"{cuts[-1] + 1:,} or more")
    else:
        labels.append(f"up to {_format(edges[0])}")
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            labels.append(f"over {_format(lo)} up to {_format(hi)}")
        labels.append(f"over {_format(edges[-1])}")
    return labels


def round_cuts(cuts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Cut points rounded to two significant figures of the spread of the data (the interquartile range), so that a scorecard
    reads "45,000" and "0.25" rather than "44,873" and "0.2514"."""
    q75, q25 = np.quantile(values, [0.75, 0.25])
    spread = (q75 - q25) or float(np.std(values))
    if not spread:
        return np.unique(cuts)
    decimals = int(np.clip(2 - np.floor(np.log10(spread)), -8, 8))
    return np.unique(np.round(cuts, decimals))


@dataclass
class FeatureBinning:
    """The bins found for one feature, the weight of evidence of each, and how to apply them to new values."""

    name: str
    kind: str                                    # "numeric" or "categorical"
    table: pd.DataFrame                          # one row per bin: label, n, bad, good, bad_rate, woe, iv
    edges: np.ndarray = field(default_factory=lambda: np.array([]))      # numeric: the cut points between bins
    groups: list[tuple] = field(default_factory=list)                    # categorical: the values in each bin
    missing_bin: int | None = None               # the bin for missing values, when the training data had any
    monotone: str | None = None

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    @property
    def n_bins(self) -> int:
        return len(self.table)

    def bin_index(self, x) -> np.ndarray:
        """The bin of each value; -1 for a category never seen in training, or a missing value when training had none."""
        values = pd.Series(x).reset_index(drop=True)
        missing = values.isna().to_numpy()
        if self.kind == "numeric":
            index = np.searchsorted(self.edges, values.fillna(0).to_numpy(dtype=float), side="left")
        else:
            lookup = {v: i for i, members in enumerate(self.groups) for v in members}
            index = values.map(lookup).fillna(-1).to_numpy(dtype=int)
        index = np.where(missing, self.missing_bin if self.missing_bin is not None else -1, index)
        return index.astype(int)

    def woe(self, x) -> np.ndarray:
        """Weight of evidence of each value. A value with no bin (an unseen category) gets 0, the average account."""
        weights = np.append(self.table["woe"].to_numpy(), 0.0)         # the last entry is what index -1 picks
        return weights[self.bin_index(x)]


def fit_binning(x, y, name: str = "x", kind: str = "numeric", monotone: str | None = None, *,
                min_share: float = config.MIN_BIN_SHARE, max_bins: int = config.MAX_BINS,
                iv_tolerance: float = config.IV_TOLERANCE, labels: dict | None = None) -> FeatureBinning:
    """Find the bins of one feature. `y` is 1 for a bad account (a default) and 0 for a good one."""
    if kind not in ("numeric", "categorical"):
        raise ValueError(f"kind must be numeric or categorical, not {kind!r}")
    values = pd.Series(x).reset_index(drop=True)
    y = np.asarray(y).astype(int)
    if len(values) != len(y):
        raise ValueError("x and y differ in length")
    if set(np.unique(y)) - {0, 1} or y.min() == y.max():
        raise ValueError("y must hold both 0 and 1")
    present = values.notna().to_numpy()
    xv, yv = values[present].reset_index(drop=True), y[present]

    edges, groups = np.array([]), []
    if kind == "numeric":
        arr = xv.to_numpy(dtype=float)
        distinct = np.unique(arr)
        if len(distinct) <= INITIAL_BINS:
            cuts = (distinct[:-1] + distinct[1:]) / 2
        else:
            cuts = round_cuts(np.quantile(arr, np.linspace(0, 1, INITIAL_BINS + 1)[1:-1]), arr)
            cuts = cuts[(cuts >= distinct[0]) & (cuts < distinct[-1])]      # every bin must be able to hold a value
        atomic = np.searchsorted(cuts, arr, side="left")                    # a value equal to a cut point goes to the bin below it
        n_atomic = len(cuts) + 1
        direction = monotone
    else:
        categories = list(pd.unique(xv))
        rates = {c: yv[(xv == c).to_numpy()].mean() for c in categories}
        ordered = sorted(categories, key=lambda c: (rates[c], str(c)))
        atomic = xv.map({c: i for i, c in enumerate(ordered)}).to_numpy(dtype=int)
        n_atomic = len(ordered)
        direction = None

    bad = np.bincount(atomic[yv == 1], minlength=n_atomic)
    good = np.bincount(atomic[yv == 0], minlength=n_atomic)
    merged = merge_neighbours(bad, good, min_share=min_share, max_bins=max_bins, iv_tolerance=iv_tolerance, monotone=direction)

    rows_bad = [int(bad[g].sum()) for g in merged]
    rows_good = [int(good[g].sum()) for g in merged]
    if kind == "numeric":
        edges = np.array([cuts[g[-1]] for g in merged[:-1]])
        integer = bool(np.all(np.mod(arr, 1) == 0))
        names = numeric_labels(edges, integer, arr.min(), arr.max())
    else:
        groups = [tuple(ordered[i] for i in g) for g in merged]
        names = [", ".join(str((labels or {}).get(v, v)) for v in members) for members in groups]

    missing_bin = None
    if not present.all():
        missing_bin = len(merged)
        rows_bad.append(int(y[~present].sum()))
        rows_good.append(int((1 - y[~present]).sum()))
        names.append("missing")
    woe, iv = woe_iv(rows_bad, rows_good)
    n = np.array(rows_bad) + np.array(rows_good)
    table = pd.DataFrame({"label": names, "n": n, "bad": rows_bad, "good": rows_good, "bad_rate": np.array(rows_bad) / n, "woe": woe, "iv": iv})
    return FeatureBinning(name=name, kind=kind, table=table, edges=edges, groups=groups, missing_bin=missing_bin, monotone=monotone)
