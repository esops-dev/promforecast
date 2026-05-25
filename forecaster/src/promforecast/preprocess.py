"""Lookback preprocessing: gap imputation, outlier scrubbing, change-point detection.

Three pre-fit cleaning steps share one pipeline so the runner can call into a
single function per series:

1. **Gap imputation** (always-on, configurable strategy). Materialises a
   regularly-sampled series at the configured step. Long gaps trip a
   fallback strategy (default: ``drop_series``).
2. **Outlier scrubbing** (opt-in). Hampel / IQR / STL-residual detector
   flags spikes; ``replace_with`` controls what replaces them.
3. **Change-point detection** (opt-in). PELT (via ``ruptures``) or a
   built-in BOCPD detector locates structural breaks; the action either
   truncates the lookback to the post-break tail or down-weights pre-break
   samples.

The ordering matters: gaps first (downstream needs aligned data), outliers
on the gap-filled series (so the detector sees a regular grid), change-point
last (so a single spike doesn't masquerade as a regime shift).

Output is a :class:`PreprocessResult` with the cleaned timestamps/values
plus a small bag of telemetry (counters by category, the largest gap, the
effective post-break lookback) the runner forwards to the exporter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import (
        BlacklistWindowConfig,
        ChangePointConfig,
        GapImputationConfig,
        OutlierConfig,
        PreprocessConfig,
    )

logger = structlog.get_logger(__name__)


# Sensitivity → method-specific threshold lookups. Hampel uses a "k * sigma"
# style threshold (smaller k flags more); IQR uses a multiplier of the
# inter-quartile range (smaller multiplier flags more); STL residuals use a
# z-score cutoff after dividing residuals by their robust scale.
_HAMPEL_K = {"low": 4.0, "medium": 3.0, "high": 2.0}
_IQR_MULT = {"low": 3.0, "medium": 1.5, "high": 1.0}
_STL_Z = {"low": 4.0, "medium": 3.0, "high": 2.0}

# CUSUM break-detection cutoff. Roughly matches a Bonferroni-corrected
# t-test at alpha = 0.01 across ~1000 candidate split positions, which is
# the typical lookback density at 5-min step / 14-day window.
_BOCPD_T_CUTOFF = 3.5

# Minimum number of usable calibration residuals needed before split-
# conformal will emit an offset. Below this the quantile is statistically
# meaningless and the runner skips calibration for this series.
_MIN_CALIB_SAMPLES = 2


@dataclass
class PreprocessResult:
    """Cleaned series plus per-step telemetry.

    ``dropped`` is set when a gap longer than ``max_gap`` triggers the
    ``drop_series`` fallback. The runner treats this as a per-series drop
    (the rest of the query continues) and surfaces it via
    ``forecast_series_dropped_total``.

    Counters are dict-of-counts so the runner can fold them into the
    exporter's per-(group, query, ...) counters without losing the sub-key
    breakdown (strategy, method).

    ``blacklisted_samples`` is a per-``reason`` counter (defaulting to
    the empty string for unreasoned windows) for samples removed by the
    ``preprocess.blacklist_windows`` list. The values flow into
    ``forecast_input_blacklisted_samples_total{reason}`` so an operator
    can confirm an annual-recurring blacklist actually fired for the
    most recent year in the lookback.
    """

    timestamps: list[datetime]
    values: list[float]
    gaps_filled: dict[str, int] = field(default_factory=dict)
    max_gap_seconds: float = 0.0
    outliers_replaced: int = 0
    change_points: int = 0
    effective_lookback_seconds: float = 0.0
    imputed_fraction: float = 0.0
    blacklisted_samples: dict[str, int] = field(default_factory=dict)
    dropped: bool = False
    drop_reason: str | None = None


def preprocess_series(
    *,
    timestamps: list[datetime],
    values: list[float],
    step: timedelta,
    config: PreprocessConfig,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on one series.

    Returns a :class:`PreprocessResult`. ``dropped: True`` means the caller
    should skip the series entirely (a multi-hour gap blew past
    ``gaps.max_gap`` with the ``drop_series`` fallback); ``dropped: False``
    means ``timestamps`` / ``values`` are safe to feed into the model fit.

    The function is pure / side-effect free apart from a structured log on
    notable events (drop, large change-point truncation). It does **not**
    touch the exporter — that's the runner's job.
    """
    if not timestamps or not values:
        return PreprocessResult(timestamps=[], values=[])
    if len(timestamps) != len(values):
        raise ValueError("preprocess: timestamps/values length mismatch")

    blacklist_counts: dict[str, int] = {}
    if config.blacklist_windows:
        timestamps, values, blacklist_counts = _drop_blacklisted(
            timestamps=timestamps,
            values=values,
            windows=config.blacklist_windows,
        )

    if not timestamps or not values:
        # The blacklist consumed every sample. Surface as a drop so the
        # caller bumps ``forecast_series_dropped_total{reason="preprocess_drop"}``
        # rather than continuing into the fit with an empty series.
        return PreprocessResult(
            timestamps=[],
            values=[],
            blacklisted_samples=blacklist_counts,
            dropped=True,
            drop_reason="blacklist_removed_all_samples",
        )

    result = _fill_gaps(timestamps=timestamps, values=values, step=step, config=config.gaps)
    result.blacklisted_samples = blacklist_counts
    if result.dropped:
        return result

    if config.outliers.enabled:
        _scrub_outliers(result, config=config.outliers)

    if config.change_point.enabled:
        _apply_change_point(result, step=step, config=config.change_point)

    if not result.effective_lookback_seconds and result.timestamps:
        span = result.timestamps[-1] - result.timestamps[0]
        result.effective_lookback_seconds = span.total_seconds()
    # Discount the blacklist-removed duration from the effective
    # lookback so the gauge reflects the actual fit-input duration
    # (a 14-day lookback with two 6-hour blacklisted windows reports
    # ``14d - 12h`` rather than the raw 14d span). Calculated from
    # ``count * step`` rather than the raw window length so a window
    # that only partially overlaps the lookback is counted accurately.
    total_blacklisted = sum(blacklist_counts.values())
    if total_blacklisted and step.total_seconds() > 0:
        deduction = total_blacklisted * step.total_seconds()
        result.effective_lookback_seconds = max(0.0, result.effective_lookback_seconds - deduction)
    return result


# ---------- blacklist windows ----------------------------------------------


def _drop_blacklisted(  # noqa: PLR0912
    *,
    timestamps: list[datetime],
    values: list[float],
    windows: list[BlacklistWindowConfig],
) -> tuple[list[datetime], list[float], dict[str, int]]:
    """Remove samples whose timestamp falls inside any blacklist window.

    Returns the surviving (timestamps, values) plus a per-``reason``
    counter of samples removed. Empty windows (zero-length absolute
    spans or unparseable patterns) are skipped silently — pydantic
    catches the gross-shape errors at boot, so anything reaching this
    helper is structurally valid.

    The windows are interpreted as half-open intervals ``[start, end)``;
    annual-recurring patterns expand to one absolute span per year
    seen in the input timestamps (so a 30-year-lookback config does
    not pay the cost of materialising 30 spans on every fit).
    """
    if not timestamps:
        return list(timestamps), list(values), {}

    # Expand recurring patterns once across the years actually present
    # in the input window so the per-sample comparison stays cheap.
    span_years = _years_present(timestamps)
    expanded: list[tuple[datetime, datetime, str]] = []
    for window in windows:
        if window.recurring == "annual":
            for year in span_years:
                pair = _parse_annual_window(
                    window.start, window.end, year, timezone_name=window.timezone
                )
                if pair is not None:
                    expanded.append((pair[0], pair[1], window.reason))
        else:
            absolute = _parse_absolute_window(window.start, window.end)
            if absolute is not None:
                expanded.append((absolute[0], absolute[1], window.reason))

    if not expanded:
        return list(timestamps), list(values), {}

    out_ts: list[datetime] = []
    out_vals: list[float] = []
    counts: dict[str, int] = {}
    for ts, val in zip(timestamps, values, strict=True):
        ts_aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
        hit_reason: str | None = None
        for start, end, reason in expanded:
            if start <= ts_aware < end:
                hit_reason = reason
                break
        if hit_reason is None:
            out_ts.append(ts)
            out_vals.append(val)
        else:
            counts[hit_reason] = counts.get(hit_reason, 0) + 1
    return out_ts, out_vals, counts


def _years_present(timestamps: list[datetime]) -> list[int]:
    """Return the distinct years present in ``timestamps``."""
    if not timestamps:
        return []
    first = timestamps[0]
    last = timestamps[-1]
    return list(range(first.year, last.year + 1))


def _parse_absolute_window(start: str, end: str) -> tuple[datetime, datetime] | None:
    """Parse an ISO-8601 ``start`` / ``end`` pair into UTC-aware datetimes.

    Returns ``None`` on a malformed input — the pydantic validator
    catches structurally-wrong shapes at config load, so reaching this
    branch implies a deeper format error worth swallowing rather than
    crashing the per-fit pipeline.
    """
    try:
        # ``fromisoformat`` is strict about the trailing ``Z`` only on
        # Python 3.11+, but ``Z`` is the standard UTC suffix the rest
        # of the codebase emits; normalise to ``+00:00`` for portability.
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    if s.tzinfo is None:
        s = s.replace(tzinfo=UTC)
    if e.tzinfo is None:
        e = e.replace(tzinfo=UTC)
    if e <= s:
        return None
    return s, e


def _parse_annual_window(
    start_pattern: str, end_pattern: str, year: int, *, timezone_name: str = "UTC"
) -> tuple[datetime, datetime] | None:
    """Materialise an annual ``MM-DDTHH:MM`` window for ``year``.

    Returns ``None`` on malformed input or an unrecognised IANA
    ``timezone_name``. The window is anchored in the operator's
    configured local zone (defaulting to UTC) so a Black-Friday-style
    rule flips at local midnight rather than UTC-midnight; the
    returned datetimes are normalised to UTC so the per-sample
    comparison in :func:`_drop_blacklisted` can stay zone-agnostic.
    """
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return None
    try:
        s_local = datetime.fromisoformat(f"{year:04d}-{start_pattern}").replace(tzinfo=tz)
        e_year = year
        # Allow the recurring window to wrap a year boundary
        # ("12-29T00:00" -> "01-02T00:00") by rolling ``end`` forward.
        e_local = datetime.fromisoformat(f"{year:04d}-{end_pattern}").replace(tzinfo=tz)
        if e_local <= s_local:
            e_year = year + 1
            e_local = datetime.fromisoformat(f"{e_year:04d}-{end_pattern}").replace(tzinfo=tz)
    except ValueError:
        return None
    # Normalise to UTC so the half-open ``[start, end)`` comparison
    # against UTC-aware sample timestamps is correct regardless of the
    # zone chosen by the operator.
    s = s_local.astimezone(UTC)
    e_candidate = e_local.astimezone(UTC)
    if e_candidate <= s:
        return None
    return s, e_candidate


# ---------- gap imputation -------------------------------------------------


def _fill_gaps(
    *,
    timestamps: list[datetime],
    values: list[float],
    step: timedelta,
    config: GapImputationConfig,
) -> PreprocessResult:
    """Materialise a regularly-spaced series at ``step`` and record gap stats.

    Walk the incoming (already sorted) timestamps and insert filler samples
    whenever the gap to the next timestamp exceeds 1.5x step. Strategies:

    * ``forward_fill``: copy the previous value forward.
    * ``linear_interp``: linearly interpolate between bracketing samples.
    * ``nan``: insert :data:`math.nan` (the model's responsibility to drop).
    * ``drop_series``: short-circuit and signal a drop to the caller.

    ``max_gap`` short-circuits to ``drop_series`` regardless of ``strategy``
    when a single gap is too long — a 4-hour outage shouldn't smuggle
    fabricated values into a 14-day forecast. ``drop_series`` *is* the
    fallback, so it never falls back to itself.
    """
    if step.total_seconds() <= 0:
        # Defensive: a non-positive step would cause an infinite loop below.
        return PreprocessResult(timestamps=list(timestamps), values=list(values))

    step_seconds = step.total_seconds()
    max_gap_seconds = max(step_seconds, config.max_gap.total_seconds())
    # Gap "boundary" — anything shorter is normal scrape jitter. 1.5x step
    # is the standard for irregular Prom data.
    gap_threshold = 1.5 * step_seconds

    out_ts: list[datetime] = [timestamps[0]]
    out_vals: list[float] = [values[0]]
    counts: dict[str, int] = {}
    largest_gap = 0.0
    imputed = 0

    for i in range(1, len(timestamps)):
        prev_t = out_ts[-1]
        cur_t = timestamps[i]
        cur_v = values[i]
        gap = (cur_t - prev_t).total_seconds()
        if gap <= gap_threshold:
            out_ts.append(cur_t)
            out_vals.append(cur_v)
            continue
        largest_gap = max(largest_gap, gap)
        if gap > max_gap_seconds and config.strategy != "drop_series":
            # Apply the fallback (always ``drop_series`` today) so a
            # very long gap doesn't get papered over by forward-fill.
            logger.info(
                "preprocess_gap_drop",
                gap_seconds=round(gap, 1),
                max_gap_seconds=max_gap_seconds,
                configured_strategy=config.strategy,
            )
            return PreprocessResult(
                timestamps=[],
                values=[],
                gaps_filled=counts,
                max_gap_seconds=largest_gap,
                dropped=True,
                drop_reason="gap_too_large",
            )
        if config.strategy == "drop_series":
            return PreprocessResult(
                timestamps=[],
                values=[],
                gaps_filled=counts,
                max_gap_seconds=largest_gap,
                dropped=True,
                drop_reason="gap_too_large",
            )
        # Generate filler timestamps stepwise so the runner sees a regular
        # grid downstream. ``+1`` because the first inserted ts is one step
        # past ``prev_t`` and the last is one step before ``cur_t``.
        n_fill = max(1, round(gap / step_seconds) - 1)
        filler_values = _fill_values(
            prev_v=out_vals[-1],
            cur_v=cur_v,
            n=n_fill,
            strategy=config.strategy,
        )
        for j, fv in enumerate(filler_values, start=1):
            out_ts.append(prev_t + step * j)
            out_vals.append(fv)
            counts[config.strategy] = counts.get(config.strategy, 0) + 1
            imputed += 1
        out_ts.append(cur_t)
        out_vals.append(cur_v)

    span_seconds = (out_ts[-1] - out_ts[0]).total_seconds() if out_ts else 0.0
    return PreprocessResult(
        timestamps=out_ts,
        values=out_vals,
        gaps_filled=counts,
        max_gap_seconds=largest_gap,
        effective_lookback_seconds=span_seconds,
        imputed_fraction=(imputed / max(1, len(out_ts))),
    )


def _fill_values(*, prev_v: float, cur_v: float, n: int, strategy: str) -> list[float]:
    """Generate ``n`` filler values between ``prev_v`` and ``cur_v``."""
    if strategy == "forward_fill":
        return [prev_v] * n
    if strategy == "linear_interp":
        if n <= 0:
            return []
        return [prev_v + (cur_v - prev_v) * ((i + 1) / (n + 1)) for i in range(n)]
    if strategy == "nan":
        return [math.nan] * n
    # ``drop_series`` is handled upstream; any other value is a config bug
    # that pydantic would have rejected.
    return [prev_v] * n


# ---------- outlier scrubbing ----------------------------------------------


def _scrub_outliers(result: PreprocessResult, *, config: OutlierConfig) -> None:
    """Detect and replace outliers in-place on ``result.values``.

    The detector returns a boolean mask; ``replace_with`` controls the
    replacement strategy. The number of replaced points is folded into
    ``result.outliers_replaced`` so the runner can bump the counter.
    """
    arr = np.asarray(result.values, dtype=float)
    if arr.size == 0:
        return
    if config.method == "hampel":
        mask = _hampel_mask(arr, k=_HAMPEL_K[config.sensitivity])
    elif config.method == "iqr":
        mask = _iqr_mask(arr, mult=_IQR_MULT[config.sensitivity])
    elif config.method == "stl_residual":
        mask = _stl_residual_mask(arr, z=_STL_Z[config.sensitivity])
    else:
        return
    n_outliers = int(mask.sum())
    if n_outliers == 0:
        return
    replacements = _build_outlier_replacements(arr, mask, strategy=config.replace_with)
    new_values = arr.copy()
    new_values[mask] = replacements[mask]
    result.values = new_values.tolist()
    result.outliers_replaced = n_outliers


def _hampel_mask(arr: NDArray[np.float64], *, k: float, window: int = 7) -> NDArray[np.bool_]:
    """Hampel filter: flag points more than ``k`` MAD-sigmas from the local median.

    Implemented with a rolling window via simple Python loops; the typical
    lookback is short (hundreds of points for 5-min step / 14-day lookback,
    ~4000 max), and the Python overhead is dominated by the surrounding fit
    cost. A numba-accelerated version is an obvious follow-up if profiling
    ever shows it's hot.

    The MAD-is-zero branch — common in piecewise-constant series like
    request counts that sit at zero for long stretches — falls back to a
    global standard deviation so a real spike is still flagged. Without
    the fallback the filter silently ignores spikes against a flat
    baseline (the most useful case for an operator-facing scrubber).
    """
    n = arr.size
    mask = np.zeros(n, dtype=bool)
    if n < window:
        return mask
    half = window // 2
    # MAD scale factor: 1.4826 makes MAD approximately equal to sigma for a Gaussian.
    mad_scale = 1.4826
    # Global fallback scale for the MAD-is-zero edge case.
    global_std = float(np.std(arr))
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[lo:hi]
        med = np.median(chunk)
        mad = np.median(np.abs(chunk - med))
        diff = abs(arr[i] - med)
        if mad == 0:
            if global_std == 0:
                continue
            if diff > k * global_std:
                mask[i] = True
            continue
        if diff > k * mad_scale * mad:
            mask[i] = True
    return mask


def _iqr_mask(arr: NDArray[np.float64], *, mult: float) -> NDArray[np.bool_]:
    """Tukey's IQR filter on the global window.

    Degenerate IQR (a series that's mostly identical) falls back to a
    standard-deviation cutoff so spike-against-flat-baseline cases — the
    most useful real-world target — don't silently slip through.
    """
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        std = float(np.std(arr))
        if std <= 0:
            return np.zeros(arr.size, dtype=bool)
        med = float(np.median(arr))
        # ``np.asarray`` pins the dtype so mypy sees a typed mask rather
        # than the ``ndarray[Any]`` numpy infers for chained comparisons.
        return np.asarray(np.abs(arr - med) > mult * std, dtype=np.bool_)
    lo = q1 - mult * iqr
    hi = q3 + mult * iqr
    return np.asarray((arr < lo) | (arr > hi), dtype=np.bool_)


def _stl_residual_mask(arr: NDArray[np.float64], *, z: float) -> NDArray[np.bool_]:
    """Flag points whose STL residual is more than ``z`` robust-sigmas from zero.

    Falls back to a Hampel filter when STL can't be applied (series too
    short, season-length heuristic fails) — better than silently emitting
    "no outliers" and letting spikes through.
    """
    n = arr.size
    # STL needs at least two full periods; pick a safe daily-style period
    # (24) when we can't infer better, halving until it fits.
    period = min(24, max(2, n // 4))
    if n < 2 * period:
        return _hampel_mask(arr, k=z)
    try:
        from statsmodels.tsa.seasonal import STL  # noqa: PLC0415

        stl = STL(arr, period=period, robust=True).fit()
        residuals = np.asarray(stl.resid, dtype=float)
    except (ValueError, RuntimeError, ImportError):
        return _hampel_mask(arr, k=z)
    # Robust scale via MAD on residuals.
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad <= 0:
        return np.zeros(n, dtype=bool)
    scale = 1.4826 * mad
    return np.asarray(np.abs(residuals) > z * scale, dtype=np.bool_)


def _build_outlier_replacements(
    arr: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    strategy: str,
    window: int = 7,
) -> NDArray[np.float64]:
    """Build a per-position replacement array for the flagged indices."""
    n = arr.size
    out = arr.copy()
    if strategy == "nan":
        out[mask] = np.nan
        return out
    if strategy == "median_neighbours":
        half = window // 2
        for raw_i in np.flatnonzero(mask):
            i = int(raw_i)
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            # Exclude the flagged point itself from the neighbour set.
            neighbour_mask: NDArray[np.bool_] = np.ones(hi - lo, dtype=np.bool_)
            neighbour_mask[i - lo] = False
            neighbours = arr[lo:hi][neighbour_mask & ~mask[lo:hi]]
            if neighbours.size:
                out[i] = float(np.median(neighbours))
            else:
                out[i] = float(np.median(arr[~mask])) if (~mask).any() else 0.0
        return out
    if strategy == "linear_interp":
        # Replace each flagged point with linear interp between the nearest
        # non-flagged samples on either side. Boundary flagged points fall
        # back to nearest-neighbour copy.
        good = ~mask
        for raw_i in np.flatnonzero(mask):
            i = int(raw_i)
            left = i - 1
            while left >= 0 and not good[left]:
                left -= 1
            right = i + 1
            while right < n and not good[right]:
                right += 1
            if left >= 0 and right < n:
                t = (i - left) / max(1, right - left)
                out[i] = float(arr[left] * (1 - t) + arr[right] * t)
            elif left >= 0:
                out[i] = float(arr[left])
            elif right < n:
                out[i] = float(arr[right])
        return out
    return out


# ---------- change-point detection -----------------------------------------


def _apply_change_point(
    result: PreprocessResult,
    *,
    step: timedelta,
    config: ChangePointConfig,
) -> None:
    """Detect breaks and either truncate or reweight ``result`` in place.

    The detector returns a single most-recent breakpoint index (we are
    interested in the *current* regime, not the historical timeline). If
    detection fails or the post-break tail is too short, the series is
    left untouched and the change-point counter is not incremented.
    """
    arr = np.asarray(result.values, dtype=float)
    if arr.size < max(20, config.min_post_break_points):
        return
    breakpoint_idx = _detect_breakpoint(arr, method=config.method)
    if breakpoint_idx is None:
        return
    post = arr.size - breakpoint_idx
    if post < config.min_post_break_points:
        return

    if config.action == "truncate_lookback":
        result.values = result.values[breakpoint_idx:]
        result.timestamps = result.timestamps[breakpoint_idx:]
        result.change_points += 1
        result.effective_lookback_seconds = (
            (result.timestamps[-1] - result.timestamps[0]).total_seconds()
            if result.timestamps
            else 0.0
        )
        # Imputed fraction is no longer meaningful for the truncated
        # window (we don't know how many imputations survived the cut).
        # Conservative re-estimate: scale by the fraction kept.
        result.imputed_fraction *= post / max(1, arr.size)
        del step  # unused after truncation
        return

    if config.action == "reweight_recent":
        # statsforecast's model API does not expose per-sample weights, so
        # we approximate down-weighting as soft truncation: keep all
        # post-break points plus the ``reweight_decay`` fraction of the
        # most-recent pre-break tail. ``decay=1.0`` is a no-op (no
        # truncation); ``decay=0.0`` collapses to a hard truncate.
        keep_pre = round(breakpoint_idx * config.reweight_decay)
        cut = max(0, breakpoint_idx - keep_pre)
        if cut > 0:
            result.values = result.values[cut:]
            result.timestamps = result.timestamps[cut:]
        result.change_points += 1
        result.effective_lookback_seconds = (
            (result.timestamps[-1] - result.timestamps[0]).total_seconds()
            if result.timestamps
            else 0.0
        )
        result.imputed_fraction *= (arr.size - cut) / max(1, arr.size)


def _detect_breakpoint(arr: NDArray[np.float64], *, method: str) -> int | None:
    """Return the index of the most-recent detected break, or ``None``.

    ``ruptures`` (PELT with an L2 cost) is the default — fast, robust, and
    avoids the need to pick a hazard rate. ``bocpd`` is a built-in
    online-style Bayesian detector kept lightweight (Gaussian likelihood,
    geometric hazard) so the feature works even when ``ruptures`` is
    unavailable.
    """
    if method == "ruptures":
        return _ruptures_breakpoint(arr)
    if method == "bocpd":
        return _bocpd_breakpoint(arr)
    return None


def _ruptures_breakpoint(arr: NDArray[np.float64]) -> int | None:
    try:
        import ruptures as rpt  # noqa: PLC0415
    except ImportError:
        return _bocpd_breakpoint(arr)
    # PELT with an L2 cost detects mean shifts; ``pen`` of 3 * log(n) is a
    # BIC-style penalty that balances false positives against missed
    # breaks for the typical 1k-4k-point lookback.
    n = arr.size
    try:
        algo = rpt.Pelt(model="l2", min_size=max(10, n // 50)).fit(arr)
        penalty = 3.0 * math.log(max(2, n))
        result = algo.predict(pen=penalty)
    except Exception:
        return None
    # ``ruptures`` returns the *end* indices of each segment ending with
    # ``n``; strip that and return the most recent break (latest index).
    breaks = [b for b in result if b < n]
    if not breaks:
        return None
    return int(breaks[-1])


def _bocpd_breakpoint(arr: NDArray[np.float64]) -> int | None:
    """Lightweight CUSUM-style detector: largest mean-shift in the tail.

    Bayesian online change-point detection is moderately involved to
    implement well; a single-break CUSUM scan over the series finds the
    same kind of structural break and is dependency-free. We pick the
    largest absolute t-statistic across split positions and accept the
    break only when it exceeds a conservative cutoff (3.5 by default,
    chosen to roughly match Bonferroni at alpha=0.01 across ~1000 candidate
    positions).
    """
    n = arr.size
    min_segment = max(10, n // 20)
    if n < 2 * min_segment:
        return None
    best_score = 0.0
    best_idx = -1
    # Pre-compute cumulative sums for O(n) sweep.
    cum = np.cumsum(arr)
    sq_cum = np.cumsum(arr * arr)
    total = cum[-1]
    sq_total = sq_cum[-1]
    for i in range(min_segment, n - min_segment):
        left_sum = cum[i - 1]
        left_sq = sq_cum[i - 1]
        right_sum = total - left_sum
        right_sq = sq_total - left_sq
        n_l = i
        n_r = n - i
        mean_l = left_sum / n_l
        mean_r = right_sum / n_r
        var_l = max(1e-9, left_sq / n_l - mean_l * mean_l)
        var_r = max(1e-9, right_sq / n_r - mean_r * mean_r)
        pooled = math.sqrt(var_l / n_l + var_r / n_r)
        if pooled <= 0:
            continue
        t = abs(mean_r - mean_l) / pooled
        if t > best_score:
            best_score = t
            best_idx = i
    if best_idx < 0 or best_score < _BOCPD_T_CUTOFF:
        return None
    return best_idx
