"""Intermittent-demand model wrappers.

The :mod:`statsforecast` library ships a family of statistical methods
designed for sparse, count-shaped signals where the underlying data is
zero most of the time and occasionally bursts to a small positive value:

* :class:`CrostonModel` — classical Croston decomposes the series into a
  demand size and an inter-demand interval and forecasts both with
  simple exponential smoothing.
* :class:`CrostonSBAModel` — Syntetos-Boylan bias-corrected Croston;
  the standard recommendation when in doubt because plain Croston is
  systematically biased upward.
* :class:`ADIDAModel` — aggregates the series to a non-intermittent
  frequency, forecasts there, then disaggregates back. Useful for very
  sparse signals where Croston struggles to estimate the inter-demand
  interval.
* :class:`IMAPAModel` — averages forecasts across multiple aggregation
  buckets, trading some bias for variance.

Bands on these methods come from ``statsforecast``'s
:class:`ConformalIntervals` bootstrap (we instantiate one whenever the
caller asks for confidence levels). Without that argument, the
underlying classes raise on any ``level=`` request — surfacing a much
more confusing error than wrapping it cleanly here.

Each wrapper accepts ``season_length`` for registry-protocol uniformity
and ignores it: intermittent-demand methods don't model seasonality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .seasonal_naive import _normalize

if TYPE_CHECKING:
    import pandas as pd


# Two windows is the statsforecast default and the documented minimum for
# the conformal bootstrap; smaller numbers degrade the quantile estimate
# below the noise floor on already-sparse intermittent series. We expose
# the knob via ``params`` so power users can raise it for noisier
# signals, but never let it drop below the floor.
_CONFORMAL_DEFAULT_WINDOWS = 2
_CONFORMAL_MIN_WINDOWS = 2


def _build_conformal_intervals(horizon: int, params: dict[str, Any]) -> object:
    """Construct a ``ConformalIntervals`` wrapper for an intermittent fit.

    The intermittent-demand models in ``statsforecast`` need an explicit
    ``prediction_intervals`` argument before they emit any band columns;
    passing ``level=[80, 95]`` against an unwrapped instance raises with
    a message that doesn't make it obvious how to fix the config. By
    materialising the bootstrap wrapper here we let operators turn on
    bands by setting the standard ``confidence_levels`` knob and don't
    leak the underlying library detail.

    ``params`` can carry ``conformal_windows`` (alias: ``n_windows``) to
    override the bootstrap size; both spellings are popped from
    ``params`` so they don't get forwarded into the model constructor
    where they would raise as unknown kwargs.
    """
    from statsforecast.utils import ConformalIntervals  # noqa: PLC0415

    n_windows = params.pop("conformal_windows", None)
    if n_windows is None:
        n_windows = params.pop("n_windows", _CONFORMAL_DEFAULT_WINDOWS)
    else:
        params.pop("n_windows", None)
    try:
        n_windows_int = int(n_windows)
    except (TypeError, ValueError):
        n_windows_int = _CONFORMAL_DEFAULT_WINDOWS
    n_windows_int = max(_CONFORMAL_MIN_WINDOWS, n_windows_int)
    return ConformalIntervals(n_windows=n_windows_int, h=horizon)


def _fit_predict_intermittent(
    *,
    model_factory: Any,
    model_name: str,
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    levels: list[int],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Fit one intermittent-demand model and normalise its output frame.

    The ``alias=model_name`` kwarg pins the column name in
    ``statsforecast``'s output to our wrapper's public name (e.g. the
    ``CrostonClassic`` class is exposed under the name ``Croston``).
    Without the pin, ``_normalize`` wouldn't find the column to rename
    into ``yhat`` and downstream emission would silently produce empty
    frames.
    """
    from statsforecast import StatsForecast  # noqa: PLC0415

    init_kwargs: dict[str, Any] = dict(params)
    init_kwargs.setdefault("alias", model_name)
    if levels:
        init_kwargs["prediction_intervals"] = _build_conformal_intervals(horizon, init_kwargs)
    sf = StatsForecast(
        models=[model_factory(**init_kwargs)],
        freq=freq,
        n_jobs=1,
    )
    fit_kwargs: dict[str, Any] = {"df": df, "h": horizon}
    if levels:
        fit_kwargs["level"] = levels
    forecast = sf.forecast(**fit_kwargs)
    return _normalize(forecast, model_name=model_name, levels=levels if levels else [])


class _IntermittentBase:
    """Shared boilerplate for the four intermittent-demand wrappers.

    ``season_length`` is accepted for protocol uniformity and ignored —
    intermittent models don't carry a seasonality concept.
    """

    name: ClassVar[str] = ""

    def __init__(self, season_length: int = 288, **params: Any) -> None:
        # ``season_length`` is part of the registry-side contract — every
        # model class accepts it. Intermittent methods don't model
        # seasonality, so we accept the kwarg and never look at it.
        self._season_length = season_length
        self._params = params

    def _statsforecast_factory(self) -> Any:
        """Return the underlying ``statsforecast`` class for this wrapper.

        Subclasses override this to point at their concrete class. Kept
        as a method rather than a class attribute so the heavy
        ``statsforecast`` import is deferred to fit time.
        """
        raise NotImplementedError

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del regressors  # intermittent models in statsforecast don't accept exogenous regressors
        return _fit_predict_intermittent(
            model_factory=self._statsforecast_factory(),
            model_name=self.name,
            df=df,
            horizon=horizon,
            freq=freq,
            levels=levels,
            params=dict(self._params),
        )


class CrostonModel(_IntermittentBase):
    """Classical Croston's method.

    Decomposes the series into demand sizes and inter-demand intervals
    and forecasts each with simple exponential smoothing. Known to be
    biased upward on average; prefer :class:`CrostonSBAModel` when in
    doubt.
    """

    name = "Croston"

    def _statsforecast_factory(self) -> Any:
        from statsforecast.models import CrostonClassic  # noqa: PLC0415

        return CrostonClassic


class CrostonSBAModel(_IntermittentBase):
    """Croston with the Syntetos-Boylan bias-correction.

    The "use Croston without thinking" default — multiplies the classical
    Croston forecast by ``1 - alpha/2`` to neutralise the upward bias.
    """

    name = "CrostonSBA"

    def _statsforecast_factory(self) -> Any:
        from statsforecast.models import CrostonSBA  # noqa: PLC0415

        return CrostonSBA


class ADIDAModel(_IntermittentBase):
    """Aggregate-Disaggregate Intermittent Demand Approach.

    Aggregates the series to a coarser, less-sparse cadence, forecasts
    there with SES, then disaggregates back to the original frequency.
    Useful when the inter-demand interval is too long for Croston to
    estimate cleanly.
    """

    name = "ADIDA"

    def _statsforecast_factory(self) -> Any:
        from statsforecast.models import ADIDA  # noqa: PLC0415

        return ADIDA


class IMAPAModel(_IntermittentBase):
    """Intermittent Multiple Aggregation Prediction Algorithm.

    Computes ADIDA-style forecasts across a range of aggregation buckets
    and averages them. Trades a little bias for lower variance —
    handy on very noisy intermittent series.
    """

    name = "IMAPA"

    def _statsforecast_factory(self) -> Any:
        from statsforecast.models import IMAPA  # noqa: PLC0415

        return IMAPA


# Convenience set consumed by the auto-select preference for
# ``data_profile: intermittent``. Derived from ``_IntermittentBase``'s
# direct subclasses so the set self-maintains: every wrapper that
# inherits the base class and sets a non-empty ``name`` lands here
# automatically. A wrapper that forgot to set ``name`` would also be
# unreachable through :func:`promforecast.models.build`, so the filter
# below silently excludes it — matching the behaviour of the registry's
# own discovery code.
INTERMITTENT_MODEL_NAMES = frozenset(
    cls.name for cls in _IntermittentBase.__subclasses__() if cls.name
)


__all__ = [
    "INTERMITTENT_MODEL_NAMES",
    "ADIDAModel",
    "CrostonModel",
    "CrostonSBAModel",
    "IMAPAModel",
]
