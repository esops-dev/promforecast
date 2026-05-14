"""Data-driven seasonality detection for ``season_length: auto``.

The static heuristic in :func:`promforecast.config._derive_season_length_from_step`
picks a daily seasonality from the step alone (``86400 / step_seconds``). That
is a sensible *default*, but it's wrong for series whose natural period is
weekly (e.g. business KPIs that follow Mon-Fri patterns) or sub-daily (e.g.
batch jobs every 4 hours). When a query opts into ``season_length: auto`` we
do better: inspect the lookback values and pick the period whose lagged
autocorrelation peaks above noise.

Algorithm:

1. Compute the autocorrelation function (ACF) up to a bounded maximum lag
   using FFT. The bound is ``min(len(values) // 2, max_lag_steps)`` so a
   pathologically long lookback can't run away with CPU.
2. Search the candidate range ``[min_lag, max_lag]`` for the strongest
   *positive* local maximum. Anti-correlation (negative) peaks land at
   half-period for sinusoidal patterns; ignoring them is what stops the
   detector from systematically picking ``period/2``.
3. **Snap to natural periods** when a peak sits close to a calendar
   boundary (hour, day, week). ACF peaks rarely land exactly on
   ``steps_per_day`` because of clock drift across the lookback window;
   snapping makes downstream model fits behave predictably.
4. If no peak crosses ``min_peak_correlation``, fall back to the supplied
   default. The default is the step-derived daily seasonality so the
   behaviour gracefully degrades into the old static heuristic.

The result is *one* season_length per series. statsforecast models like
MSTL accept multiple seasonalities, but the public ``ForecastModel``
protocol exposes a single integer; multi-seasonality is a future model-pack
concern and not in scope here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

# Bounds on the candidate seasonality range. Lower bound 2 because
# season_length=1 is "no seasonality" and statsforecast already handles
# that case internally; ACF at lag 1 is autoregression, not seasonality.
_MIN_SEASON_LENGTH = 2

# Cap the FFT-based ACF computation. For step=5m and lookback=14d that's
# 4032 samples; we don't need to search past half that for a meaningful
# period detection. Bounding the max lag also keeps a misconfigured
# long lookback from blowing up CPU on the per-series detection.
_MAX_LAG_STEPS = 2048

# Cap the FFT *input* length too. The lag cap keeps the search bounded,
# but the FFT itself scales with the full input. We trim to the most
# recent ``_MAX_INPUT_LENGTH`` samples — 8x the max lag, enough for
# four full cycles of the longest detectable period. A misconfigured
# 90-day, 1-minute lookback (~130k samples) would otherwise allocate
# ~1 MiB per detection and dominate the per-series fit budget for no
# additional accuracy.
_MAX_INPUT_LENGTH = 8 * _MAX_LAG_STEPS

# A peak must clear this autocorrelation magnitude to count. ACF values
# in [0, 1] (we use absolute value to be robust to anti-correlation at
# half-period). 0.2 is a conservative noise floor — strong daily/weekly
# patterns easily clear 0.5; ambient noise rarely exceeds 0.2 over many
# lags. Below the floor we fall back rather than pick a spurious peak.
_MIN_PEAK_CORRELATION = 0.2

# Tolerance for snapping a peak to a natural calendar boundary. ACF
# peaks drift by 1-2 steps over a multi-week lookback because of
# real-world timing jitter; we snap when the candidate is within this
# fraction of the natural period.
_NATURAL_SNAP_TOLERANCE = 0.05


@dataclass(frozen=True)
class SeasonalityDetection:
    """Outcome of a single seasonality detection pass.

    ``period`` is the chosen season_length. ``source`` is one of:

    * ``"detected"`` — ACF found a strong peak; ``correlation`` is its strength.
    * ``"snapped"`` — ACF peak snapped to a natural calendar period
      (hour, day, week). ``correlation`` is the original peak strength.
    * ``"fallback"`` — no peak cleared the noise floor; ``period`` is the
      caller's fallback value and ``correlation`` is ``NaN``.
    * ``"too_short"`` — series too short to evaluate; ``period`` is the
      caller's fallback.
    """

    period: int
    source: str
    correlation: float


def detect_season_length(
    values: list[float] | tuple[float, ...],
    step: timedelta,
    fallback: int,
    *,
    min_peak_correlation: float = _MIN_PEAK_CORRELATION,
) -> SeasonalityDetection:
    """Pick a season_length for ``values`` using FFT autocorrelation.

    Returns the detected period plus diagnostics. Callers that don't care
    about diagnostics can read ``.period`` and ignore the rest.

    A bottom floor of ``_MIN_SEASON_LENGTH`` is enforced because
    statsforecast's seasonal models can't operate on season_length < 2.
    The caller's ``fallback`` is *also* clamped to that floor for the same
    reason.
    """
    fallback_clamped = max(_MIN_SEASON_LENGTH, fallback)
    n = len(values)
    # Need at least 4 full cycles at the minimum lag to compute a meaningful
    # ACF estimate. With _MIN_SEASON_LENGTH=2 that's 8 samples — anything
    # less and we can't beat noise.
    if n < 4 * _MIN_SEASON_LENGTH:
        return SeasonalityDetection(
            period=fallback_clamped, source="too_short", correlation=float("nan")
        )

    import numpy as np  # noqa: PLC0415

    # Trim to the most recent _MAX_INPUT_LENGTH samples before the FFT —
    # the lag cap alone doesn't bound FFT cost; the full input does. The
    # most recent slice is preferable to a head/tail trim because real
    # series often have ramp-up artifacts at the start of the lookback
    # window (gauge initialisation, scrape gaps, restarts) that don't
    # carry useful seasonal signal.
    if n > _MAX_INPUT_LENGTH:
        values = values[-_MAX_INPUT_LENGTH:]
        n = _MAX_INPUT_LENGTH
    arr = np.asarray(values, dtype=float)
    # Guard against constant / near-constant series: ACF is undefined when
    # variance is zero, and would otherwise spuriously return lag=1.
    if not np.isfinite(arr).all() or np.allclose(arr.std(), 0.0):
        return SeasonalityDetection(
            period=fallback_clamped, source="too_short", correlation=float("nan")
        )

    max_lag = min(n // 2, _MAX_LAG_STEPS)
    if max_lag <= _MIN_SEASON_LENGTH:
        return SeasonalityDetection(
            period=fallback_clamped, source="too_short", correlation=float("nan")
        )

    acf = _autocorrelation(arr, max_lag)
    peak_lag, peak_value = _strongest_peak(acf, min_lag=_MIN_SEASON_LENGTH, max_lag=max_lag)

    if peak_value < min_peak_correlation:
        return SeasonalityDetection(
            period=fallback_clamped, source="fallback", correlation=float("nan")
        )

    snapped = _snap_to_natural_period(peak_lag, step)
    if snapped is not None and snapped != peak_lag:
        return SeasonalityDetection(period=snapped, source="snapped", correlation=peak_value)
    return SeasonalityDetection(
        period=max(_MIN_SEASON_LENGTH, peak_lag), source="detected", correlation=peak_value
    )


def _autocorrelation(arr: object, max_lag: int) -> object:
    """Biased ACF up to ``max_lag``, normalised by the lag-0 value.

    FFT-based computation keeps the cost at O(N log N) for the whole curve
    instead of O(N * max_lag) for the naive per-lag approach. For a
    14-day, 5-minute-step series (~4000 samples) that's the difference
    between ~50 ms and ~1 s per series — important when many series
    detect concurrently.
    """
    import numpy as np  # noqa: PLC0415

    a = np.asarray(arr, dtype=float)
    a = a - a.mean()
    n = len(a)
    # Zero-pad to the next power of two to keep the FFT efficient and
    # avoid circular-correlation artifacts at high lags.
    fft_n = 1 << math.ceil(math.log2(2 * n))
    fft = np.fft.rfft(a, n=fft_n)
    power = fft * np.conj(fft)
    acf_full = np.fft.irfft(power, n=fft_n)[:n].real
    if acf_full[0] == 0:
        return np.zeros(max_lag + 1, dtype=float)
    return acf_full[: max_lag + 1] / acf_full[0]


def _strongest_peak(acf: object, *, min_lag: int, max_lag: int) -> tuple[int, float]:
    """Return ``(lag, value)`` for the highest positive local-maximum within bounds.

    A "peak" is a lag whose ACF is strictly greater than its neighbours.
    Only positive ACF values count: a series with period ``P`` shows a
    *negative* peak at lag ``P/2`` (anti-correlation) and a *positive*
    peak at lag ``P``. If we accepted absolute values, ``P/2`` would
    win because it's the first qualifying local max in the curve and
    ties with ``P`` in magnitude — so the chosen seasonality would
    consistently be half the real period.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(acf, dtype=float)
    candidates = arr[min_lag : max_lag + 1]
    if candidates.size < 3:  # noqa: PLR2004
        return 0, 0.0
    # Local maxima of the (signed) ACF, then filter to positive values.
    left = candidates[1:-1] > candidates[:-2]
    right = candidates[1:-1] > candidates[2:]
    positive = candidates[1:-1] > 0
    peak_mask = left & right & positive
    if not peak_mask.any():
        return 0, 0.0
    peak_indices = np.flatnonzero(peak_mask) + 1  # offset back into ``candidates``
    peak_strengths = candidates[peak_indices]
    best = int(np.argmax(peak_strengths))
    return int(peak_indices[best]) + min_lag, float(peak_strengths[best])


def _snap_to_natural_period(peak_lag: int, step: timedelta) -> int | None:
    """Snap ``peak_lag`` to hour/day/week if it's close to one of those.

    Returns the snapped period when a snap occurred, else ``None``. The
    tolerance is ``_NATURAL_SNAP_TOLERANCE`` (5% by default) so a detected
    peak at lag 286 with step=5m snaps to 288 (a clean day); 250 doesn't.
    """
    step_seconds = max(1, int(step.total_seconds()))
    naturals: list[int] = []
    for seconds in (3600, 86_400, 7 * 86_400):
        candidate = seconds // step_seconds
        if candidate >= _MIN_SEASON_LENGTH:
            naturals.append(candidate)
    if not naturals:
        return None
    for natural in naturals:
        if natural == 0:
            continue
        if abs(peak_lag - natural) / natural <= _NATURAL_SNAP_TOLERANCE:
            return natural
    return None


__all__ = [
    "SeasonalityDetection",
    "detect_season_length",
]
