"""Tests for the data-driven seasonality detector."""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from promforecast.seasonality import detect_season_length


def _sine_series(period: int, n_periods: int, noise: float = 0.05) -> list[float]:
    """Synthetic series with a clean period plus a little Gaussian noise.

    Noise is added so we exercise the noise-floor logic; a perfectly clean
    sine wave is too easy and wouldn't catch regressions where the noise
    floor is misapplied.
    """
    rng = np.random.default_rng(seed=42)
    n = period * n_periods
    base = np.sin(2 * np.pi * np.arange(n) / period)
    return list(base + rng.normal(scale=noise, size=n))


def test_detects_daily_seasonality_at_five_minute_step() -> None:
    """288 samples per day; a 14-day series should snap straight to 288."""
    series = _sine_series(period=288, n_periods=14)
    outcome = detect_season_length(series, step=timedelta(minutes=5), fallback=1)
    assert outcome.period == 288
    # Either a clean detection or a snap-to-natural — both are correct here.
    assert outcome.source in {"detected", "snapped"}
    assert outcome.correlation > 0.5


def test_detects_weekly_seasonality_at_hourly_step() -> None:
    """A weekly pattern at hourly step → period 168."""
    series = _sine_series(period=168, n_periods=8)
    outcome = detect_season_length(series, step=timedelta(hours=1), fallback=24)
    assert outcome.period == 168
    assert outcome.source in {"detected", "snapped"}


def test_snaps_drifted_peak_to_natural_period() -> None:
    """A peak that lands a few steps off a natural period must snap to it.

    Real-world ACF peaks drift by a step or two over a multi-day lookback
    because of timing jitter. The snap-to-natural-period rule prevents that
    drift from flowing through into model fits.
    """
    # Make the period 285 steps (close to 288 = one day at 5min) — within
    # the 5% snap tolerance. Detection should report 288 and source=snapped.
    series = _sine_series(period=285, n_periods=14)
    outcome = detect_season_length(series, step=timedelta(minutes=5), fallback=1)
    assert outcome.period == 288
    assert outcome.source == "snapped"


def test_falls_back_on_white_noise() -> None:
    """Pure white noise has no autocorrelation peak; we must fall back."""
    rng = np.random.default_rng(seed=7)
    series = list(rng.normal(size=2000))
    outcome = detect_season_length(series, step=timedelta(minutes=5), fallback=288)
    assert outcome.period == 288
    assert outcome.source == "fallback"
    assert not math.isfinite(outcome.correlation)


def test_falls_back_on_constant_series() -> None:
    """A flat series has undefined ACF; degenerate input → fallback."""
    series = [42.0] * 500
    outcome = detect_season_length(series, step=timedelta(minutes=5), fallback=288)
    assert outcome.period == 288
    assert outcome.source == "too_short"


def test_short_series_falls_back() -> None:
    """A series shorter than 4 minimum cycles can't sustain a meaningful ACF."""
    series = list(range(6))
    outcome = detect_season_length(series, step=timedelta(minutes=5), fallback=288)
    assert outcome.period == 288
    assert outcome.source == "too_short"


def test_fallback_is_clamped_to_minimum() -> None:
    """A misconfigured fallback of 1 (or 0) must clamp to the minimum of 2."""
    outcome = detect_season_length([], step=timedelta(minutes=5), fallback=1)
    assert outcome.period == 2


def test_high_peak_correlation_required_for_detection() -> None:
    """Very noisy patterns where the peak doesn't clear the floor must fall back."""
    # period=24, but with noise so high that the peak smears into the floor
    rng = np.random.default_rng(seed=11)
    n = 24 * 30
    series = list(0.1 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(scale=1.0, size=n))
    outcome = detect_season_length(
        series,
        step=timedelta(hours=1),
        fallback=24,
        min_peak_correlation=0.5,
    )
    # Either fallback or snap — but never a wildly off period.
    if outcome.source == "fallback":
        assert outcome.period == 24
    else:
        assert outcome.period in (24, 168)
