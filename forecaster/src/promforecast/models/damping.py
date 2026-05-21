"""Shared trend-damping post-processing for the damped model variants.

Used by :class:`AutoARIMADampedModel` and :class:`AutoETSDampedModel` so
the ``damping_parameter`` knob behaves identically across the two models.
The transform decomposes the predicted curve into a linear trend plus an
oscillation (seasonal / cyclic) component, replaces the linear part with
a geometric sum that decays at rate ``phi``, and shifts the confidence
bands by the same delta so they ride along with the bent forecast
without changing width.

The decomposition is deliberate: a naive "multiply yhat by phi^k" would
destroy seasonal shape. Fitting and replacing only the trend keeps the
forecast's shape intact while bending the long-horizon extrapolation
toward a level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# 0.98 is the capacity-planning rule of thumb cited in ``docs/models.md``:
# at horizon=288 (24h at 5m step) it caps the extrapolated trend at
# roughly the asymptote ``phi / (1 - phi) = 49`` trend units — well
# under the undamped 288 — while leaving short-horizon forecasts within
# a few percent of the linear extrapolation. Tight enough to keep
# disk-fill forecasts honest without flattening a genuine multi-day
# growth trend.
DEFAULT_DAMPING_PARAMETER = 0.98

# Damping fits a line to the forecast curve, which requires at least two
# rows; with a single row there is no trend to bend.
_MIN_ROWS_FOR_DAMPING = 2


def apply_damping_to_forecast(
    forecast: pd.DataFrame,
    *,
    phi: float,
    levels: list[int],
) -> pd.DataFrame:
    """Return ``forecast`` with damped ``yhat`` and band columns.

    ``phi`` ∈ (0, 1] controls the damping. ``phi == 1.0`` is a no-op so
    callers can disable damping by configuration without a code branch.
    Anything outside ``(0, 1]`` is also a no-op — the underlying ETS /
    ARIMA fit already produced a valid curve, and an out-of-range phi
    should not silently corrupt it (the schema validator rejects the
    config before we get here, but defensive in case of programmatic
    construction).

    The forecast frame is expected in the normalised ``yhat`` /
    ``yhat_lower_<level>`` / ``yhat_upper_<level>`` shape produced by
    :func:`promforecast.models.seasonal_naive._normalize`. Other columns
    pass through unchanged.
    """
    import numpy as np  # noqa: PLC0415

    if not (0.0 < phi <= 1.0) or phi == 1.0:
        return forecast
    if forecast.empty or "yhat" not in forecast.columns:
        return forecast

    yhat = forecast["yhat"].to_numpy(dtype=float)
    n = yhat.size
    if n < _MIN_ROWS_FOR_DAMPING:
        return forecast

    h = np.arange(1, n + 1, dtype=float)
    # ``np.polyfit`` returns highest-degree first.
    slope, intercept = np.polyfit(h, yhat, 1)
    linear = intercept + slope * h
    residual = yhat - linear
    # Geometric sum: sum_{k=1..n} phi^k = phi * (1 - phi^n) / (1 - phi).
    damped_steps = slope * phi * (1.0 - np.power(phi, h)) / (1.0 - phi)
    damped_linear = intercept + damped_steps
    damped_yhat = damped_linear + residual
    shift = damped_yhat - yhat

    out = forecast.copy()
    out["yhat"] = damped_yhat
    for level in levels:
        for col in (f"yhat_lower_{level}", f"yhat_upper_{level}"):
            if col in out.columns:
                out[col] = out[col].to_numpy(dtype=float) + shift
    return out
