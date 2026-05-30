"""Misc pure helpers extracted from the runner monolith.

Each function here either reshapes a pandas frame, formats a label
value, or wraps an upstream helper to keep the runner's call-sites
short. Nothing here depends on a live :class:`~promforecast.runner.Runner`
instance.
"""

from __future__ import annotations

import functools
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .. import point_factory as point_factory_mod
from ..accuracy import backtest
from ..config import ModelSpec
from ..point_factory import BandSummary

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from ..models import ForecastModel


def _series_key_from_labels(labels: dict[str, str]) -> str:
    """Stable per-series key derived from the label set.

    Used to address per-series state in the drift cache and band-coverage
    tracker. Skips ``__name__`` (Prometheus's reserved metric-name
    pseudo-label) so two queries that return the same labelset on a
    different metric share state appropriately — both forecast the same
    underlying signal.

    The key is ``"k1=v1,k2=v2"`` with sorted keys so equal label
    dictionaries produce the same key regardless of iteration order.
    """
    items = sorted((k, v) for k, v in labels.items() if k != "__name__")
    return ",".join(f"{k}={v}" for k, v in items)


def _format_band_window(window: timedelta) -> str:
    """Render a ``timedelta`` window as a Prometheus-style label."""
    return point_factory_mod.format_window(window.total_seconds())


# Suffixes that the runner / point_factory emits on forecast-derived
# metrics. ``_query_id_from_metric`` matches the longest one first so an
# id like ``api_forecast_count`` (a legal Prometheus identifier) is not
# mis-routed by an internal ``_forecast`` substring.
_FORECAST_METRIC_SUFFIXES: tuple[str, ...] = (
    "_forecast_contribution_ratio",
    "_forecast_time_to_threshold_seconds",
    "_forecast_time_to_drain_seconds",
    "_forecast_error_budget_remaining_seconds",
    "_forecast_error_budget_exhausted_in_seconds",
    "_forecast_error_budget_burn_rate",
    "_forecast_growth_rate",
    "_forecast_will_breach",
    "_forecast_conditional_lower",
    "_forecast_conditional_upper",
    "_forecast_conditional",
    "_forecast_component",
    "_forecast_quantile",
    "_forecast_lower",
    "_forecast_upper",
    "_forecast",
)


# Suffixes indexed by length (longest first). Each ``length`` maps to a
# ``frozenset`` so the longest-suffix probe in ``_query_id_from_metric`` is an
# O(1) hash lookup per distinct length rather than an O(N_suffixes) linear
# scan. With today's 15-suffix set there are 13 distinct lengths, so a
# no-match call costs at most 13 hash lookups instead of 15 ``endswith``
# scans — but the realistic call shape from
# ``_inherit_prior_for_skipped_queries`` calls the helper thousands of times
# per refresh on a tiny set of distinct metric names, so the
# ``functools.lru_cache`` wrapper below collapses the repeat-call cost to a
# single cache hit. Cache size is bounded by ``N_queries x N_suffixes``
# (≤ 300 in any realistic config), well under the ``maxsize`` cap.
_SUFFIXES_BY_LENGTH: dict[int, frozenset[str]] = {}
for _suffix in _FORECAST_METRIC_SUFFIXES:
    _SUFFIXES_BY_LENGTH[len(_suffix)] = _SUFFIXES_BY_LENGTH.get(len(_suffix), frozenset()) | {
        _suffix
    }
_SUFFIX_LENGTHS_DESC: tuple[int, ...] = tuple(sorted(_SUFFIXES_BY_LENGTH.keys(), reverse=True))


@functools.lru_cache(maxsize=2048)
def _query_id_from_metric(metric_name: str) -> str:
    """Pull the query ``id`` back out of a forecast metric name.

    Matches the known emission suffix set rather than ``rfind("_forecast")``
    so a query id that legitimately contains the word ``forecast`` (e.g.
    ``api_forecast_count``) round-trips correctly. Lookup is bounded by the
    number of distinct suffix lengths (longest-first probe of a
    length-indexed ``frozenset``) and memoised so the hot path through
    ``_inherit_prior_for_skipped_queries`` pays one hash lookup per repeat
    call.
    """
    name_len = len(metric_name)
    for length in _SUFFIX_LENGTHS_DESC:
        if name_len <= length:
            continue
        if metric_name[-length:] in _SUFFIXES_BY_LENGTH[length]:
            return metric_name[:-length]
    return ""


def _resolve_spec(specs: list[ModelSpec], name: str) -> ModelSpec:
    """Look up a configured ``ModelSpec`` by name; defaults to params=``{}``.

    The fallback model is allowed to live outside ``defaults.models`` (a
    SeasonalNaive safety net for an AutoARIMA-only setup is the canonical
    case). We synthesise an empty-params spec so the fallback runs with
    stock model defaults rather than failing the lookup.
    """
    for spec in specs:
        if spec.name == name:
            return spec
    return ModelSpec(name=name)


def _combine_ensemble(
    component_forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
    levels: list[int],
) -> pd.DataFrame:
    """Build a single forecast frame from inverse-MAPE-weighted components.

    Aligns on ``unique_id`` + ``ds``. Each numeric column is the weighted
    sum across components present at that row; missing columns (a model
    that didn't emit a level we requested) contribute weight 0 so the
    ensemble doesn't quietly inherit one component's missing band.
    """
    import pandas as pd  # noqa: PLC0415

    base_cols = ["unique_id", "ds"]
    value_cols = ["yhat"] + [
        f"yhat_{kind}_{level}" for kind in ("lower", "upper") for level in levels
    ]

    import numpy as np  # noqa: PLC0415

    accumulator: pd.DataFrame | None = None
    weight_accumulator: dict[str, np.ndarray[Any, Any]] | None = None
    for name, weight in weights.items():
        df = component_forecasts.get(name)
        if df is None or df.empty:
            continue
        if accumulator is None or weight_accumulator is None:
            accumulator = df[base_cols].reset_index(drop=True).copy()
            weight_accumulator = {col: np.zeros(len(df), dtype=float) for col in value_cols}
            for col in value_cols:
                accumulator[col] = 0.0
        for col in value_cols:
            if col not in df.columns:
                continue
            # Switch to numpy arrays for the accumulation: pandas would
            # try to align on the (matching) index every iteration, which
            # both wastes work and confuses pandas-stubs about the
            # resulting dtype.
            current = accumulator[col].to_numpy(dtype=float)
            accumulator[col] = current + df[col].to_numpy(dtype=float) * weight
            weight_accumulator[col] = weight_accumulator[col] + weight
    if accumulator is None or weight_accumulator is None:
        # No usable components — return an empty frame the caller can detect.
        return pd.DataFrame(columns=[*base_cols, *value_cols])
    # Convert weight sums back to plain Series for the divide-by-weight step;
    # rows where total weight is 0 retain NaN rather than dividing by zero.
    for col in value_cols:
        denom = pd.Series(weight_accumulator[col])
        accumulator[col] = accumulator[col] / denom.where(denom > 0, other=float("nan"))
    return accumulator


def _run_backtest(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon_steps: int,
    freq: str,
    levels: list[int],
    regressors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    return backtest(
        model=model,
        df=df,
        horizon_steps=horizon_steps,
        freq=freq,
        levels=levels,
        regressors=regressors,
    )


def _pandas_freq(step: timedelta) -> str:
    seconds = int(step.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _summarise_backtest(
    predictions: pd.DataFrame,
    test: pd.DataFrame,
    levels: list[int],
) -> tuple[float, float, dict[int, BandSummary]] | None:
    """Extract the trailing (actual, yhat, per-level band) tuple, or None.

    Returns ``None`` if either frame is empty or missing the ``y``/``yhat``
    columns; non-finite cells are tolerated and surface downstream as NaN.
    """
    if predictions.empty or test.empty:
        return None
    try:
        last_actual = float(test["y"].iloc[-1])
        last_yhat = float(predictions["yhat"].iloc[-1])
    except (KeyError, IndexError, ValueError):
        return None
    bands: dict[int, BandSummary] = {}
    for level in levels:
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col not in predictions.columns or hi_col not in predictions.columns:
            continue
        try:
            lower = float(predictions[lo_col].iloc[-1])
            upper = float(predictions[hi_col].iloc[-1])
        except (ValueError, IndexError):
            continue
        bands[level] = BandSummary(lower=lower, upper=upper)
    return last_actual, last_yhat, bands


def _run_fit(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    levels: list[int],
    regressors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return model.fit_predict(
        df=df, horizon=horizon, freq=freq, levels=levels, regressors=regressors
    )


# Floor for relative-metric denominators across the per-horizon
# revision + disagreement helpers below; matches the
# :mod:`promforecast.revision` scale floor.
_RELATIVE_METRIC_FLOOR = 1e-9
# Minimum number of distinct finite values required to compute the
# disagreement coefficient of variation. Two is the smallest mathematically
# meaningful count; below that there's no spread to measure.
_MIN_DISAGREEMENT_VALUES = 2


def _extract_per_horizon_yhat(forecast: Any, horizon_specs: list[Any]) -> dict[str, float]:
    """Return ``{horizon_label: yhat_value}`` from a forecast frame.

    ``horizon_specs`` is the list produced by
    :func:`promforecast.accuracy.horizons_from_durations`. Each entry's
    ``steps`` is the 1-indexed step count for that horizon — we read
    row ``steps - 1`` of ``forecast`` for that horizon's yhat. Rows
    past the end of the frame and non-finite values are skipped.
    """
    import math  # noqa: PLC0415

    if forecast is None or len(forecast) == 0:
        return {}
    out: dict[str, float] = {}
    try:
        yhat_col = forecast["yhat"]
    except (KeyError, TypeError):
        return out
    n = len(forecast)
    for spec in horizon_specs:
        idx = int(spec.steps) - 1
        if idx < 0 or idx >= n:
            continue
        try:
            value = float(yhat_col.iloc[idx])
        except (KeyError, IndexError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        out[spec.label] = value
    return out


def _coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation (stddev / |mean|) across ``values``.

    Used as the model-disagreement score: 0 = perfect agreement, higher
    = more disagreement. Returns ``NaN`` when fewer than 2 finite
    values are present (no disagreement to measure) or when the mean
    is too close to zero for the relative metric to be meaningful.
    """
    import math  # noqa: PLC0415

    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < _MIN_DISAGREEMENT_VALUES:
        return float("nan")
    mean = sum(finite) / len(finite)
    denom = abs(mean)
    if denom < _RELATIVE_METRIC_FLOOR:
        return float("nan")
    variance = sum((v - mean) ** 2 for v in finite) / len(finite)
    return math.sqrt(variance) / denom


def _estimate_group_memory_bytes(group_name: str, exporter: Any) -> float:
    """Best-effort per-group snapshot memory estimate.

    Sums ``sys.getsizeof`` over every list-valued field on the snapshot
    so the resulting gauge tracks "which group's snapshot is dominant"
    without pulling in psutil. Fields are discovered via
    :func:`dataclasses.fields` so a new emission family that adds a
    ``list`` attribute to :class:`GroupSnapshot` is accounted for
    without an edit here — the hardcoded-list shape this replaced
    silently undercounted the snapshot every time a new family
    landed. The number is still a recursive undercount (Python's
    object graph isn't fully traversed) and an overcount when point
    objects are shared across snapshots; intent is comparative, not
    exact. Falls back to 0.0 when the snapshot is missing.
    """
    import dataclasses  # noqa: PLC0415
    import sys  # noqa: PLC0415

    snapshot = exporter.snapshot_for(group_name)
    if snapshot is None:
        return 0.0
    total = sys.getsizeof(snapshot)
    if not dataclasses.is_dataclass(snapshot):
        return float(total)
    for snap_field in dataclasses.fields(snapshot):
        items = getattr(snapshot, snap_field.name, None)
        if not isinstance(items, list) or not items:
            continue
        total += sys.getsizeof(items)
        for item in items:
            total += sys.getsizeof(item)
    return float(total)
