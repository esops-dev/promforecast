"""Count-distribution model wrappers (Poisson, negative-binomial, INGARCH).

The continuous-trend models (AutoARIMA, AutoETS, ...) assume a Gaussian
error and produce symmetric bands that routinely extend below zero on
integer-valued metrics — request counts, queue depths, error counts.
The count-aware emission safeguards (``emission.clip_non_negative`` /
``round_to_integer``) paper over the symptom by clipping and rounding the
output, but the underlying fit is still mis-specified: a Gaussian model
fit to a low-rate Poisson signal has the wrong band shape regardless of
how the output is post-processed.

These three wrappers model the integer nature of the data directly so the
predicted bands respect the distribution — they cannot extend below zero
and they are asymmetric where the count distribution is:

* :class:`PoissonModel` — the conditional mean follows a saturated
  seasonal model (the per-cycle-position sample mean, which is the
  maximum-likelihood rate estimate under a seasonal Poisson), and the
  bands are the Poisson quantiles at that rate. Use when the variance is
  approximately equal to the mean (the Poisson assumption).
* :class:`NegativeBinomialModel` — same seasonal mean, but the bands come
  from a negative-binomial distribution whose overdispersion is estimated
  by method of moments. Use when the variance exceeds the mean
  (overdispersed counts — the common real-world case).
* :class:`INGARCHModel` — an integer-valued GARCH(1,1) (a.k.a. Poisson
  autoregression): the conditional rate ``lambda_t = omega + alpha *
  y_{t-1} + beta * lambda_{t-1}`` is fit by maximum likelihood. Use when
  the count series shows autocorrelation (bursty arrivals, self-exciting
  processes) that a static seasonal rate misses.

**Decision rule.** Poisson when ``variance ≈ mean``; negative binomial
when ``variance > mean``; INGARCH when the series shows autocorrelation.
The auto-selector prefers this family for ``data_profile: count`` while
keeping the continuous models as a safety net — see
``docs/reference/models.md``.

Each wrapper ignores the ``regressors`` argument (``uses_regressors =
False``); count regression with exogenous inputs is a future extension.
Bands are produced for the requested ``levels`` only; an empty ``levels``
list yields a point forecast with no band columns, matching the other
wrappers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import structlog

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

# Per-level ``(lower, upper)`` band arrays. Typed with ``Any`` element
# arrays to sidestep numpy's invariant shape generics — the same
# pragmatic choice the point factory makes for its per-step value arrays.
_Bands = dict[int, "tuple[Any, Any]"]

# A rate floor keeps the Poisson/negative-binomial quantile functions and
# the INGARCH log-likelihood well-defined when a seasonal position (or the
# whole series) is all zeros. It is small enough not to perturb a genuine
# low rate and large enough to avoid ``log(0)`` in the INGARCH NLL.
_RATE_FLOOR = 1e-6
# INGARCH needs enough history for the maximum-likelihood recursion to be
# meaningful; below this we fall back to the seasonal-mean Poisson path
# rather than fit three parameters on a handful of points.
_INGARCH_MIN_POINTS = 12
# Upper bound on the method-of-moments overdispersion estimate. A handful
# of large residuals on a low-rate series can drive the per-point ratio
# ``(resid^2 - mu) / mu^2`` to enormous values; without a cap the
# negative-binomial bands degenerate to a near-useless [0, very-large]
# span. Real overdispersion rarely exceeds single digits, so a generous
# ceiling rejects only the noise-driven pathological case.
_MAX_OVERDISPERSION = 100.0


def _two_sided_tail_probs(level: int) -> tuple[float, float]:
    """Return the (lower, upper) quantile probabilities for a band level.

    A nominal ``level`` (e.g. 80) describes a two-sided central interval,
    so the lower edge is the ``(1 - level/100) / 2`` quantile and the
    upper edge its mirror. Level 80 -> (0.1, 0.9); level 95 -> (0.025,
    0.975).
    """
    alpha = 1.0 - max(0, min(100, level)) / 100.0
    return alpha / 2.0, 1.0 - alpha / 2.0


def _seasonal_mean_rates(
    y: NDArray[np.float64], season_length: int, horizon: int
) -> NDArray[np.float64]:
    """Project the per-cycle-position sample mean across the horizon.

    Each forecast step inherits the mean of the in-sample observations
    that share its clock position (index modulo ``season_length``) — the
    maximum-likelihood rate of a saturated seasonal Poisson model.
    Positions with no in-sample data fall back to the global mean so the
    projection is always defined.
    """
    n = y.size
    sl = max(1, season_length)
    global_mean = float(np.mean(y)) if n else 0.0
    sums = np.zeros(sl, dtype=float)
    counts = np.zeros(sl, dtype=float)
    positions = np.arange(n) % sl
    np.add.at(sums, positions, y)
    np.add.at(counts, positions, 1.0)
    means = np.where(counts > 0.0, sums / np.maximum(counts, 1.0), global_mean)
    future_positions = np.arange(n, n + horizon) % sl
    rates: NDArray[np.float64] = np.maximum(means[future_positions], 0.0)
    return rates


def _future_timestamps(df: pd.DataFrame, horizon: int, freq: str) -> Any:
    """Build ``horizon`` future timestamps on the configured ``freq`` grid.

    Mirrors the forward ``ds`` index ``statsforecast`` synthesises for the
    other wrappers — anchored at the last input timestamp and stepped by
    the configured ``freq`` (e.g. ``"5min"``). Deriving the grid from
    ``freq`` rather than the empirical input spacing keeps the count
    wrappers byte-consistent with every other model, so anything that reads
    the forecast ``ds`` (the sink curve, capacity ETAs) sees the same grid
    regardless of which model produced the row.
    """
    import pandas as pd  # noqa: PLC0415

    ds = pd.to_datetime(df["ds"])
    # A fully-empty input has no anchor timestamp; fall back to the epoch so
    # the frame is still well-formed (the values are all zero anyway).
    last = ds.iloc[-1] if len(ds) else pd.Timestamp("1970-01-01")
    return pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]


def _assemble_frame(
    *,
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    yhat: NDArray[np.float64],
    bands: _Bands,
) -> pd.DataFrame:
    """Assemble the protocol-shaped forecast frame from arrays.

    ``bands`` maps each level to its ``(lower, upper)`` arrays. The output
    carries ``unique_id``, ``ds``, ``yhat`` and a
    ``yhat_lower_<level>`` / ``yhat_upper_<level>`` pair per level — the
    same columns ``_normalize`` produces for the statsforecast wrappers.
    """
    import pandas as pd  # noqa: PLC0415

    unique_id = df["unique_id"].iloc[0] if len(df) else "s"
    out: dict[str, Any] = {
        "unique_id": [unique_id] * horizon,
        "ds": _future_timestamps(df, horizon, freq),
        "yhat": yhat,
    }
    for level, (lower, upper) in bands.items():
        out[f"yhat_lower_{level}"] = lower
        out[f"yhat_upper_{level}"] = upper
    return pd.DataFrame(out)


def _poisson_bands(rates: NDArray[np.float64], levels: list[int]) -> _Bands:
    """Per-level Poisson quantile bands at the given per-step rates."""
    from scipy import stats  # noqa: PLC0415

    safe = np.maximum(rates, _RATE_FLOOR)
    bands: _Bands = {}
    for level in levels:
        p_lo, p_hi = _two_sided_tail_probs(level)
        lower = np.asarray(stats.poisson.ppf(p_lo, safe), dtype=float)
        upper = np.asarray(stats.poisson.ppf(p_hi, safe), dtype=float)
        bands[level] = (np.maximum(lower, 0.0), np.maximum(upper, 0.0))
    return bands


def _estimate_overdispersion(y: NDArray[np.float64], fitted_mean: NDArray[np.float64]) -> float:
    """Method-of-moments overdispersion ``alpha`` for ``var = mu + alpha*mu^2``.

    Averaged over the in-sample points where the fitted mean is positive;
    clamped at zero so an under-dispersed (variance < mean) series
    degrades cleanly to the Poisson limit rather than producing a negative
    dispersion.
    """
    mask = fitted_mean > _RATE_FLOOR
    if not np.any(mask):
        return 0.0
    mu = fitted_mean[mask]
    resid_sq = (y[mask] - mu) ** 2
    per_point = (resid_sq - mu) / (mu**2)
    alpha = float(np.mean(per_point))
    if not math.isfinite(alpha):
        return 0.0
    return min(_MAX_OVERDISPERSION, max(0.0, alpha))


def _negbinom_bands(rates: NDArray[np.float64], alpha: float, levels: list[int]) -> _Bands:
    """Per-level negative-binomial quantile bands at the given rates.

    Uses the ``(var = mu + alpha*mu^2)`` parameterisation: ``size = 1 /
    alpha`` and ``prob = 1 / (1 + alpha*mu)``. When ``alpha`` collapses to
    the Poisson limit the bands fall back to the Poisson quantiles so the
    two families agree at the boundary.
    """
    if alpha <= _RATE_FLOOR:
        return _poisson_bands(rates, levels)
    from scipy import stats  # noqa: PLC0415

    safe = np.maximum(rates, _RATE_FLOOR)
    size = 1.0 / alpha
    prob = 1.0 / (1.0 + alpha * safe)
    bands: _Bands = {}
    for level in levels:
        p_lo, p_hi = _two_sided_tail_probs(level)
        lower = np.asarray(stats.nbinom.ppf(p_lo, size, prob), dtype=float)
        upper = np.asarray(stats.nbinom.ppf(p_hi, size, prob), dtype=float)
        bands[level] = (np.maximum(lower, 0.0), np.maximum(upper, 0.0))
    return bands


def _ingarch_recursion(
    y: NDArray[np.float64], omega: float, alpha: float, beta: float
) -> NDArray[np.float64]:
    """Conditional-rate path ``lambda_t = omega + alpha*y_{t-1} + beta*lambda_{t-1}``."""
    n = y.size
    lam = np.empty(n, dtype=float)
    lam[0] = max(float(np.mean(y)) if n else _RATE_FLOOR, _RATE_FLOOR)
    for t in range(1, n):
        lam[t] = omega + alpha * y[t - 1] + beta * lam[t - 1]
    clamped: NDArray[np.float64] = np.maximum(lam, _RATE_FLOOR)
    return clamped


def _fit_ingarch(y: NDArray[np.float64]) -> tuple[float, float, float] | None:
    """Maximum-likelihood ``(omega, alpha, beta)`` for an INGARCH(1,1) fit.

    Minimises the Poisson negative log-likelihood of the conditional-rate
    recursion under non-negativity bounds and the stationarity constraint
    ``alpha + beta < 1`` (enforced via the bound ceilings). Returns
    ``None`` when the optimiser fails to converge so the caller can fall
    back to the seasonal-mean Poisson path.
    """
    from scipy import optimize  # noqa: PLC0415

    def _nll(params: NDArray[np.float64]) -> float:
        omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
        lam = _ingarch_recursion(y, omega, alpha, beta)
        return float(np.sum(lam - y * np.log(lam)))

    mean_y = max(float(np.mean(y)), _RATE_FLOOR)
    x0 = np.array([mean_y * 0.5, 0.3, 0.3], dtype=float)
    bounds = [(_RATE_FLOOR, None), (0.0, 0.95), (0.0, 0.95)]
    try:
        res = optimize.minimize(_nll, x0, method="L-BFGS-B", bounds=bounds)
    except Exception:  # any optimiser failure routes to the seasonal-mean fallback
        logger.warning("ingarch_fit_failed")
        return None
    if not bool(res.success):
        return None
    omega, alpha, beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
    if alpha + beta >= 1.0:  # non-stationary; the forecast would diverge
        return None
    return omega, alpha, beta


def _ingarch_forecast_rates(
    y: NDArray[np.float64], params: tuple[float, float, float], horizon: int
) -> NDArray[np.float64]:
    """Project the conditional rate across the horizon from a fitted INGARCH.

    The one-step-ahead rate uses the last observed value; subsequent steps
    substitute ``E[y] = lambda`` so the projection converges to the
    stationary mean ``omega / (1 - alpha - beta)``.
    """
    omega, alpha, beta = params
    lam_hist = _ingarch_recursion(y, omega, alpha, beta)
    rates = np.empty(horizon, dtype=float)
    rates[0] = omega + alpha * float(y[-1]) + beta * float(lam_hist[-1])
    for h in range(1, horizon):
        rates[h] = omega + (alpha + beta) * rates[h - 1]
    clamped: NDArray[np.float64] = np.maximum(rates, _RATE_FLOOR)
    return clamped


class _CountBase:
    """Shared boilerplate for the count-distribution wrappers.

    Subclasses implement :meth:`_forecast` returning the ``(yhat, bands)``
    pair; the base handles input coercion, the empty-history guard, and
    frame assembly. ``season_length`` cascades from ``defaults`` like every
    other model; the count wrappers use it to project the seasonal rate.
    """

    name: ClassVar[str] = ""
    uses_regressors: ClassVar[bool] = False

    def __init__(self, season_length: int = 288, **params: Any) -> None:
        self._season_length = season_length
        self._params = params

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del regressors  # count wrappers don't consume exogenous regressors
        y = np.clip(np.asarray(df["y"].to_numpy(dtype=float)), 0.0, None)
        # Non-finite inputs (gap-imputed NaNs that slipped through) are
        # dropped so the rate estimate stays well-defined.
        y = y[np.isfinite(y)]
        if y.size == 0:
            zeros: NDArray[np.float64] = np.zeros(horizon, dtype=float)
            empty_bands: _Bands = {level: (zeros, zeros) for level in levels}
            return _assemble_frame(df=df, horizon=horizon, freq=freq, yhat=zeros, bands=empty_bands)
        yhat, bands = self._forecast(y=y, horizon=horizon, levels=levels)
        return _assemble_frame(df=df, horizon=horizon, freq=freq, yhat=yhat, bands=bands)

    def _forecast(
        self, *, y: NDArray[np.float64], horizon: int, levels: list[int]
    ) -> tuple[NDArray[np.float64], _Bands]:
        raise NotImplementedError


class PoissonModel(_CountBase):
    """Seasonal-rate Poisson model with Poisson quantile bands."""

    name = "Poisson"

    def _forecast(
        self, *, y: NDArray[np.float64], horizon: int, levels: list[int]
    ) -> tuple[NDArray[np.float64], _Bands]:
        rates = _seasonal_mean_rates(y, self._season_length, horizon)
        return rates, _poisson_bands(rates, levels)


class NegativeBinomialModel(_CountBase):
    """Seasonal-rate model with negative-binomial (overdispersed) bands."""

    name = "NegativeBinomial"

    def _forecast(
        self, *, y: NDArray[np.float64], horizon: int, levels: list[int]
    ) -> tuple[NDArray[np.float64], _Bands]:
        rates = _seasonal_mean_rates(y, self._season_length, horizon)
        # The method-of-moments dispersion estimate needs the fitted mean
        # at each *observed* index (its clock-position mean), so build the
        # position means once and index them back to 0..n-1.
        sl = max(1, self._season_length)
        positions = np.arange(y.size) % sl
        sums = np.zeros(sl, dtype=float)
        counts = np.zeros(sl, dtype=float)
        np.add.at(sums, positions, y)
        np.add.at(counts, positions, 1.0)
        global_mean = float(np.mean(y))
        means = np.where(counts > 0.0, sums / np.maximum(counts, 1.0), global_mean)
        fitted = means[positions]
        alpha = _estimate_overdispersion(y, fitted)
        return rates, _negbinom_bands(rates, alpha, levels)


class INGARCHModel(_CountBase):
    """Integer-valued GARCH(1,1) — Poisson autoregression with MLE bands.

    Falls back to the seasonal-mean Poisson projection when the series is
    too short for a meaningful three-parameter fit or the optimiser does
    not converge to a stationary solution, so a deployment can never end
    up with a diverging forecast.
    """

    name = "INGARCH"

    def _forecast(
        self, *, y: NDArray[np.float64], horizon: int, levels: list[int]
    ) -> tuple[NDArray[np.float64], _Bands]:
        params = _fit_ingarch(y) if y.size >= _INGARCH_MIN_POINTS else None
        if params is None:
            rates = _seasonal_mean_rates(y, self._season_length, horizon)
        else:
            rates = _ingarch_forecast_rates(y, params, horizon)
        return rates, _poisson_bands(rates, levels)


# Convenience set consumed by the auto-select preference for
# ``data_profile: count``. Self-maintains from ``_CountBase``'s direct
# subclasses, mirroring ``INTERMITTENT_MODEL_NAMES``: a new count wrapper
# lands in the preference whitelist by inheriting the base and setting a
# non-empty ``name``.
COUNT_MODEL_NAMES = frozenset(cls.name for cls in _CountBase.__subclasses__() if cls.name)


__all__ = [
    "COUNT_MODEL_NAMES",
    "INGARCHModel",
    "NegativeBinomialModel",
    "PoissonModel",
]
