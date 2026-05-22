"""Cold-start sibling-borrowing for newly-launched services.

When a new service goes live, every forecasting model needs more history
than it has — `min_points` typically demands a fortnight of data — so
the runner would drop the series with ``reason="min_points"`` and the
operator gets no forecast on day one. That is exactly the moment
operators need a forecast: capacity surprises happen at launch.

This module synthesises a forecast for that case. It queries a
configurable sibling set (peer services running the same workload, peer
instances of the same service, etc.), normalises each sibling to its
own mean so absolute magnitudes don't dominate the prior, takes the
per-step median across siblings to produce a smoothed seasonal *shape*,
and scales the shape back to the target's recent average. The result
is a SeasonalNaive-style forecast — no model is fit on the target's
own data.

The forecast frame produced here matches the normalised shape every
:mod:`promforecast.models` wrapper emits (``unique_id`` / ``ds`` /
``yhat``) so the runner's existing emission path consumes it without
special-casing. Confidence bands are omitted — the prior is a single
median curve and we don't have enough samples to claim a calibrated
spread.

The runner caps the ``forecast_quality_score`` for cold-start fits to
:attr:`promforecast.config.ColdStartConfig.max_quality_score` so they
cannot pass strict alert gates (``forecast_quality_score > 0.6``) on
fabricated data.

**Explicit non-goal**: this is not hierarchical forecasting
(bottom-up / top-down reconciliation). The borrowed shape is a *prior*,
not a constrained joint forecast. True reconciliation is a v2 concern.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import pandas as pd

    from .source import PromSource, SeriesFrame

logger = structlog.get_logger(__name__)


# Cold-start name on the emitted forecast point. The runner's snapshot
# rendering picks up the literal "cold_start" model label so dashboards
# can route around fabricated series without parsing a separate field.
COLD_START_MODEL_NAME = "ColdStart"

# A sibling whose lookback mean is numerically zero (or close enough)
# can't contribute to the normalised shape — dividing by its mean would
# explode the prior. We treat anything below this magnitude as
# "constant-zero" and drop it from the median pool.
_NEAR_ZERO_MEAN = 1e-12


@dataclass(frozen=True)
class ColdStartResult:
    """Outcome of attempting a cold-start fit for one series.

    ``forecast`` is the pandas frame the runner consumes (same shape as
    a model wrapper's output, sans bands). ``sibling_count`` is the
    number of usable sibling series the borrow query produced. The
    runner records this on /metrics so an operator can see when
    cold-start started borrowing from a thinner set than expected
    (e.g. siblings got renamed and the borrow query returns one less
    each refresh).

    ``shape_dispersion`` is the median absolute deviation of the
    sibling shapes from the median prior, in normalised units. Zero
    means every sibling tracked the same shape exactly; larger values
    mean the borrow set was disagreeing with itself. The runner uses
    this as the raw cold-start quality signal — high agreement plus
    enough siblings = higher (but still capped) quality.
    """

    forecast: pd.DataFrame
    sibling_count: int
    shape_dispersion: float = 0.0


def _sibling_usable_mean(series: Sequence[float], season_length: int) -> float | None:
    """Return the finite mean of ``series`` if it qualifies as a usable sibling.

    A sibling is "usable" when it (a) has at least one full seasonal
    cycle of samples, (b) carries at least one finite value, and (c)
    has a non-near-zero finite mean. Returns ``None`` when any of
    those fail; callers treat ``None`` as "drop this sibling".

    Centralising the test here keeps :func:`normalize_sibling_shape`,
    :func:`synthesise_cold_start_forecast`'s ``usable_siblings`` count,
    and :func:`_shape_dispersion` from drifting apart — earlier versions
    of this module computed the filter three different ways and the
    "how many siblings are usable" number didn't always agree.
    """
    if len(series) < season_length:
        return None
    finite = [v for v in series if math.isfinite(v)]
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    if abs(mean) < _NEAR_ZERO_MEAN:
        return None
    return mean


def normalize_sibling_shape(
    siblings: Sequence[Sequence[float]],
    season_length: int,
) -> list[float] | None:
    """Median-normalised per-step prior across sibling series.

    Each sibling is divided by its own mean so absolute magnitudes
    don't dominate (a small service plus a large service of the same
    shape produce the same prior). The mean is computed over the
    sibling's full lookback rather than over the seasonal window so a
    rapidly-growing peer doesn't smear its own trend into the prior.

    The function returns a vector of length ``season_length``: the
    median of every sibling's value at position ``i % season_length``
    across the lookback. Returns ``None`` when no sibling has enough
    points to contribute a single full seasonal cycle.

    Constant siblings (mean ~ 0) are dropped from the median because
    dividing by zero would produce NaN / inf; if every sibling is
    constant the function returns ``None`` so the caller can drop the
    series rather than emit a degenerate forecast.
    """
    if season_length < 1:
        return None
    bucket: list[list[float]] = [[] for _ in range(season_length)]
    contributed = 0
    for series in siblings:
        mean = _sibling_usable_mean(series, season_length)
        if mean is None:
            continue
        for i, value in enumerate(series):
            if not math.isfinite(value):
                continue
            bucket[i % season_length].append(value / mean)
        contributed += 1
    if contributed == 0:
        return None
    shape: list[float] = []
    for slot in bucket:
        if not slot:
            return None
        shape.append(float(statistics.median(slot)))
    return shape


def count_usable_siblings(
    siblings: Sequence[Sequence[float]],
    season_length: int,
) -> int:
    """Number of siblings that pass :func:`_sibling_usable_mean`.

    Exposed at module level so the runner can compute the sibling
    headroom once after prefetching, instead of re-deriving it inside
    :func:`synthesise_cold_start_forecast` per series.
    """
    return sum(1 for s in siblings if _sibling_usable_mean(s, season_length) is not None)


def build_cold_start_forecast(
    prior_shape: Sequence[float],
    *,
    anchor_value: float,
    horizon: int,
    step: timedelta,
    start: datetime | None = None,
    series_id: str = "s",
) -> pd.DataFrame:
    """Render the borrowed shape into a forecast frame for the runner.

    ``prior_shape`` is the unit-mean median curve from
    :func:`normalize_sibling_shape`. We re-scale it by ``anchor_value``
    (the target's own recent average) so the forecast lives on the
    target's magnitude scale, then repeat the shape across ``horizon``
    steps starting at ``start`` (or "now" when omitted).

    The output frame carries ``yhat`` only — no bands. The runner is
    responsible for adding the ``cold_start="true"`` label on emission.
    """
    import pandas as pd  # noqa: PLC0415

    if horizon < 1 or not prior_shape:
        return pd.DataFrame(columns=["unique_id", "ds", "yhat"])
    base_ts = start if start is not None else datetime.now(tz=UTC)
    period = len(prior_shape)
    scale = anchor_value if math.isfinite(anchor_value) else 0.0
    ds = [base_ts + step * (i + 1) for i in range(horizon)]
    yhat = [prior_shape[i % period] * scale for i in range(horizon)]
    return pd.DataFrame(
        {
            "unique_id": [series_id] * horizon,
            "ds": pd.to_datetime(ds, utc=True),
            "yhat": yhat,
        }
    )


async def fetch_siblings(
    *,
    source: PromSource,
    promql: str,
    lookback: timedelta,
    step: timedelta,
    query_timeout_seconds: float,
) -> list[SeriesFrame]:
    """Run the sibling discovery PromQL and return the resulting frames.

    Wrapped here so the runner's cold-start path doesn't have to
    duplicate the datetime / step boilerplate that the regressor and
    main-fetch paths already use. Errors propagate to the caller — the
    runner treats any failure as "no cold-start prior available" and
    falls back to the regular ``min_points`` drop.
    """
    import asyncio  # noqa: PLC0415

    end = datetime.now(tz=UTC)
    start = end - lookback
    step_seconds = max(1, int(step.total_seconds()))
    return await asyncio.wait_for(
        source.query_range(promql, start, end, step_seconds),
        timeout=query_timeout_seconds,
    )


def synthesise_cold_start_forecast(
    *,
    target_values: Sequence[float],
    sibling_frames: Sequence[SeriesFrame],
    season_length: int,
    horizon: int,
    step: timedelta,
    min_siblings: int,
    last_ts: datetime | None,
) -> ColdStartResult | None:
    """Top-level helper that ties the prior + render steps together.

    Returns ``None`` when not enough siblings carried usable data —
    that's the signal to the caller to fall back to the standard
    ``min_points`` drop. The threshold is enforced *after* filtering
    siblings for finite mean and minimum length so an operator who
    asks for 3 siblings but happens to query 5 of which 3 are
    degenerate gets a clean "not enough" rather than a forecast built
    on the last good one.

    ``last_ts`` is used as the anchor for the forecast's first
    timestamp; defaults to "now" if absent.
    """
    siblings = [list(frame.values) for frame in sibling_frames]
    prior = normalize_sibling_shape(siblings, season_length=season_length)
    if prior is None:
        return None
    usable_siblings = count_usable_siblings(siblings, season_length=season_length)
    if usable_siblings < min_siblings:
        logger.info(
            "cold_start_insufficient_siblings",
            usable=usable_siblings,
            required=min_siblings,
            queried=len(sibling_frames),
        )
        return None
    finite_target = [v for v in target_values if math.isfinite(v)]
    anchor = sum(finite_target) / len(finite_target) if finite_target else 0.0
    forecast = build_cold_start_forecast(
        prior_shape=prior,
        anchor_value=anchor,
        horizon=horizon,
        step=step,
        start=last_ts,
    )
    dispersion = _shape_dispersion(siblings, prior=prior, season_length=season_length)
    return ColdStartResult(
        forecast=forecast,
        sibling_count=usable_siblings,
        shape_dispersion=dispersion,
    )


def cold_start_quality_score(
    *,
    sibling_count: int,
    shape_dispersion: float,
    min_siblings: int,
    cap: float,
) -> float:
    """Map a cold-start outcome to a quality score capped at ``cap``.

    Two signals fold into the raw score:

    * **Sibling abundance**: ``min(1, sibling_count / max(min_siblings, 1))``.
      Hitting the operator's configured ``min_siblings`` is the
      contract — at that point abundance is full. Having more siblings
      doesn't give extra credit; meeting fewer would have aborted
      cold-start upstream.
    * **Shape agreement**: ``1 / (1 + dispersion)``. Zero dispersion
      (every sibling tracks the median exactly) maps to 1.0; large
      disagreement decays smoothly toward 0. The function never
      crosses 0 so a noisy sibling set can still contribute a non-
      zero quality, but the cap below keeps it from running away.

    The product is then clamped at ``cap`` (the configured
    ``max_quality_score``) so cold-start forecasts cannot pass strict
    alert gates like ``forecast_quality_score > 0.6`` regardless of
    how well the siblings agreed.

    Returns 0 when ``cap`` is non-positive (operator disabled cold-
    start quality entirely) or any input is non-finite.
    """
    if cap <= 0 or not math.isfinite(cap):
        return 0.0
    if not math.isfinite(shape_dispersion):
        return 0.0
    floor = max(min_siblings, 1)
    abundance = min(1.0, sibling_count / float(floor))
    agreement = 1.0 / (1.0 + max(0.0, shape_dispersion))
    raw = abundance * agreement
    return max(0.0, min(cap, raw * cap))


def _shape_dispersion(
    siblings: Sequence[Sequence[float]],
    *,
    prior: Sequence[float],
    season_length: int,
) -> float:
    """Median absolute deviation of sibling shapes from the median prior.

    Each sibling is renormalised to unit-mean and aligned to the prior
    by position. We compute ``|normalised_sibling_i - prior[i mod L]|``
    for every (sibling, step) and return the global median. ``0`` means
    every sibling tracked the prior exactly; larger values mean the
    siblings disagreed with each other. Used by
    :func:`cold_start_quality_score` as the agreement signal.

    Returns ``0.0`` when no sibling is usable; the caller has already
    bailed by then via :func:`normalize_sibling_shape`, but defensive
    in case the contract drifts.
    """
    if season_length < 1 or not prior:
        return 0.0
    deviations: list[float] = []
    for series in siblings:
        mean = _sibling_usable_mean(series, season_length)
        if mean is None:
            continue
        for i, value in enumerate(series):
            if not math.isfinite(value):
                continue
            normalised = value / mean
            deviations.append(abs(normalised - prior[i % season_length]))
    if not deviations:
        return 0.0
    return float(statistics.median(deviations))


__all__ = [
    "COLD_START_MODEL_NAME",
    "ColdStartResult",
    "build_cold_start_forecast",
    "cold_start_quality_score",
    "count_usable_siblings",
    "fetch_siblings",
    "normalize_sibling_shape",
    "synthesise_cold_start_forecast",
]
