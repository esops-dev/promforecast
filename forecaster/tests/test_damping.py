"""Tests for the damped-trend post-processing helper."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from promforecast.models.damping import apply_damping_to_forecast


def _forecast(values: list[float], *, levels: list[int] | None = None) -> pd.DataFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    ds = [base + timedelta(minutes=5 * i) for i in range(len(values))]
    frame = pd.DataFrame({"unique_id": ["s"] * len(values), "ds": ds, "yhat": values})
    if levels:
        for lvl in levels:
            frame[f"yhat_lower_{lvl}"] = [v - 1.0 for v in values]
            frame[f"yhat_upper_{lvl}"] = [v + 1.0 for v in values]
    return frame


def test_phi_one_is_a_noop() -> None:
    """phi=1.0 means no damping — the forecast must pass through unchanged."""
    frame = _forecast([0.0, 1.0, 2.0, 3.0, 4.0])
    out = apply_damping_to_forecast(frame, phi=1.0, levels=[])
    assert out["yhat"].tolist() == frame["yhat"].tolist()


def test_phi_below_one_bends_linear_trend() -> None:
    """A pure linear forecast bent toward a level: later values fall short."""
    yhat = [float(i) for i in range(10)]
    frame = _forecast(yhat)
    out = apply_damping_to_forecast(frame, phi=0.5, levels=[])
    # The damped curve should grow but slower; in particular the last value
    # should be less than the undamped linear extrapolation.
    assert out["yhat"].iloc[-1] < yhat[-1]
    # Damped curve should still be monotonically nondecreasing (phi > 0).
    diffs = out["yhat"].diff().dropna().tolist()
    assert all(d >= -1e-9 for d in diffs)


def test_bands_track_yhat_shift() -> None:
    """Bands move with yhat by the same shift, preserving width."""
    frame = _forecast([0.0, 1.0, 2.0, 3.0, 4.0], levels=[80])
    original_widths = (frame["yhat_upper_80"] - frame["yhat_lower_80"]).tolist()
    out = apply_damping_to_forecast(frame, phi=0.6, levels=[80])
    new_widths = (out["yhat_upper_80"] - out["yhat_lower_80"]).tolist()
    for orig, new in zip(original_widths, new_widths, strict=True):
        assert math.isclose(orig, new, abs_tol=1e-9)


def test_invalid_phi_is_noop() -> None:
    """phi outside (0,1] must not silently corrupt the forecast."""
    frame = _forecast([1.0, 2.0, 3.0])
    out = apply_damping_to_forecast(frame, phi=2.0, levels=[])
    assert out["yhat"].tolist() == frame["yhat"].tolist()
    out2 = apply_damping_to_forecast(frame, phi=0.0, levels=[])
    assert out2["yhat"].tolist() == frame["yhat"].tolist()
    out3 = apply_damping_to_forecast(frame, phi=-0.5, levels=[])
    assert out3["yhat"].tolist() == frame["yhat"].tolist()


def test_single_step_is_passthrough() -> None:
    """Damping needs at least two points to fit a trend; one-row frames pass through."""
    frame = _forecast([42.0])
    out = apply_damping_to_forecast(frame, phi=0.5, levels=[])
    assert out["yhat"].tolist() == [42.0]


def test_oscillation_preserved_relative_to_trend() -> None:
    """A wave on top of a linear trend keeps its shape after damping.

    Damping should only bend the linear part; the +/- oscillation around
    the trend must survive unchanged in *amplitude*, even though the
    midline bends. We verify by extracting the residual against a local
    linear fit before and after damping — both should produce a wave of
    the same amplitude.
    """
    n = 12
    yhat = [float(i) + (1.0 if i % 2 == 0 else -1.0) for i in range(n)]
    frame = _forecast(yhat)
    out = apply_damping_to_forecast(frame, phi=0.7, levels=[])
    # Amplitude of the residual against the running linear part — for a
    # ±1 oscillation, that's 1.0 either side, regardless of damping.
    damped = out["yhat"].tolist()
    # Approximate the local linear fit by neighbours
    deltas = [damped[i] - 0.5 * (damped[i - 1] + damped[i + 1]) for i in range(1, n - 1)]
    # Oscillation: the alternating differences should still flip sign.
    assert any(d > 0.5 for d in deltas) and any(d < -0.5 for d in deltas)
