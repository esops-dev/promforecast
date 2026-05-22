"""Forecast-vs-previous-forecast drift telemetry.

After every successful fit, the runner compares the new forecast vector to
the previous fit's forecast vector for the same series and emits a single
``forecast_drift_score{id, group, model, method}`` gauge plus a per-(group,
id) ``forecast_drift_alerts_total`` counter. The intuition: a sudden shift
in the prediction itself — independent of any observed actuals — is a
regime change worth surfacing before deviation alerts catch up.

Drift is deliberately scored *against the prediction curve*, not against
actuals. ``forecast_deviation_*`` already covers "reality differs from
prediction"; drift covers "the prediction is itself unstable".

Two scoring methods today:

* ``method: mape`` (default) — symmetric mean absolute percentage change
  between corresponding forecast steps. Operator-friendly: "the curve
  moved 25%" reads directly.
* ``method: wasserstein`` — 1-D Wasserstein (earth-mover) distance over
  the curves treated as empirical distributions, normalised by
  ``mean(|previous|)`` so it stays dimension-free. Picks up
  distributional shape changes that pointwise MAPE can miss (e.g. the
  same total but redistributed across the horizon).

This module only computes; the runner caches the previous curve per
(group, query, series-key, model) and feeds the comparison through here.

Sibling-project label alignment: the ``method`` label on
``forecast_drift_score`` is diagnostic — it describes *how* the score
was computed, not the underlying signal — and is not expected to appear
on any sibling ``promanomaly`` metric. Joins between the two projects
should be on ``id``, ``group``, and ``model`` and ignore ``method`` on
the forecaster side.
"""

from __future__ import annotations

import math
from typing import Literal

DriftMethod = Literal["mape", "wasserstein"]


def compute_drift(
    previous: list[float],
    current: list[float],
    *,
    method: DriftMethod = "mape",
) -> float:
    """Return a non-negative drift score between two forecast curves.

    Returns ``NaN`` when the curves cannot be compared — empty input or
    no finite samples in either curve. Operators treat ``NaN`` as
    "no prior fit" and skip the alert check; the gauge is not emitted
    in that case so dashboards aren't polluted with sentinel rows.

    The MAPE form is symmetric: it divides the absolute change by
    ``(|previous| + |current|) / 2`` so a curve that shifted from 0 to
    100 reports the same score as one that shifted from 100 to 0
    (otherwise zero baselines would explode to infinity asymmetrically).

    Length mismatch is handled per-method: the per-step MAPE form
    compares the leading ``min(len(previous), len(current))`` steps
    (the rest has no counterpart in the other curve); the Wasserstein
    form is distributional and resamples both curves to the longer
    length so the mass of every point contributes.
    """
    if not previous or not current:
        return float("nan")
    if method == "wasserstein":
        return _wasserstein([float(v) for v in previous], [float(v) for v in current])
    n = min(len(previous), len(current))
    prev = [float(v) for v in previous[:n]]
    cur = [float(v) for v in current[:n]]
    return _smape_curve(prev, cur)


def _smape_curve(prev: list[float], cur: list[float]) -> float:
    """Symmetric per-step MAPE averaged across the curve."""
    samples: list[float] = []
    for p, c in zip(prev, cur, strict=True):
        if not (math.isfinite(p) and math.isfinite(c)):
            continue
        denom = (abs(p) + abs(c)) / 2.0
        if denom == 0:
            # Both zero — perfect agreement on a flat-zero forecast.
            samples.append(0.0)
            continue
        samples.append(abs(p - c) / denom)
    if not samples:
        return float("nan")
    return float(sum(samples) / len(samples))


def _wasserstein(prev: list[float], cur: list[float]) -> float:
    """Normalised 1-D Wasserstein distance (sorted-difference integral).

    Wasserstein-1 between two 1-D samples of equal size is the L1
    distance between their sorted vectors divided by N. We scale by
    ``mean(|prev|)`` so the result is dimension-free; operators
    interpret the value as a relative distance ("the curve's mass moved
    by 30% of its typical magnitude").

    Mismatched finite counts (e.g. ``prev`` has 4 finite values and
    ``cur`` has 6) would otherwise produce a biased estimate by silently
    truncating the longer sorted vector. We instead linearly interpolate
    the shorter curve up to the longer length so both sums cover the
    full mass of each distribution.
    """
    prev_sorted = sorted(v for v in prev if math.isfinite(v))
    cur_sorted = sorted(v for v in cur if math.isfinite(v))
    if not prev_sorted or not cur_sorted:
        return float("nan")
    n = max(len(prev_sorted), len(cur_sorted))
    prev_resampled = _resample_sorted(prev_sorted, n)
    cur_resampled = _resample_sorted(cur_sorted, n)
    diffs = [abs(p - c) for p, c in zip(prev_resampled, cur_resampled, strict=True)]
    distance = sum(diffs) / n
    abs_prev = [abs(v) for v in prev_resampled]
    scale = sum(abs_prev) / n if abs_prev else 0.0
    if scale == 0:
        # All-zero previous curve. The distance is whatever the current
        # curve sums to; treat absolute distance as the unscaled score.
        return distance
    return distance / scale


def _resample_sorted(values: list[float], n: int) -> list[float]:
    """Linearly interpolate a sorted vector to length ``n``.

    With ``len(values) == n`` returns ``values`` unchanged. With shorter
    inputs we sample ``n`` evenly-spaced positions along the existing
    sorted curve and linearly interpolate between neighbours, which is
    the standard way to compare empirical distributions of different
    sample sizes via Wasserstein-1.
    """
    m = len(values)
    if m == n:
        return list(values)
    if m == 1:
        return [values[0]] * n
    out: list[float] = []
    last_idx = m - 1
    for i in range(n):
        pos = i * last_idx / (n - 1)
        lo = int(pos)
        hi = min(lo + 1, last_idx)
        frac = pos - lo
        out.append(values[lo] * (1.0 - frac) + values[hi] * frac)
    return out


class DriftCache:
    """In-memory cache of the most recent forecast curve per series/model.

    Keys are ``(group, query, series_key, model)`` tuples. The runner
    pulls the previous curve out at fit time, scores it against the new
    one, and writes the new curve back. The cache is per-replica and
    in-memory — a process restart resets it, which is correct: drift
    against a curve we never produced is meaningless.

    The cache is bounded indirectly by the live config: the runner
    prunes keys whose ``(group, query)`` no longer appears in the
    config on every reload, so an operator-driven cleanup keeps the
    high-water mark in check.
    """

    def __init__(self) -> None:
        self._curves: dict[tuple[str, str, str, str], list[float]] = {}

    def get(self, *, group: str, query: str, series_key: str, model: str) -> list[float] | None:
        return self._curves.get((group, query, series_key, model))

    def put(
        self,
        *,
        group: str,
        query: str,
        series_key: str,
        model: str,
        curve: list[float],
    ) -> None:
        self._curves[(group, query, series_key, model)] = curve

    def prune_to(self, live_pairs: set[tuple[str, str]]) -> None:
        """Drop cached curves whose (group, query) no longer exists."""
        self._curves = {k: v for k, v in self._curves.items() if (k[0], k[1]) in live_pairs}
