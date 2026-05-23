"""AutoARIMADamped wrapper around statsforecast.

Variant of :class:`AutoARIMAModel` that applies a *damped* trend in
post-processing. Classical ARIMA models don't carry a damping parameter
the way Holt's method does, so we run a standard AutoARIMA fit and then
fold the predicted curve through a geometric trend-damping pass — fitting
a line to the forecast values, replacing the linear extrapolation with
a damped (geometric) sum, and shifting the confidence bands by the same
delta so they follow the bent forecast without changing width.

The single tunable ``damping_parameter`` (alias: ``phi``) controls the
strength of the bend: 1.0 is no damping (equivalent to AutoARIMA),
smaller values bend the forecast toward a level. Defaults match the
roadmap-documented capacity-planning rule of thumb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .damping import DEFAULT_DAMPING_PARAMETER, apply_damping_to_forecast
from .seasonal_naive import _normalize

if TYPE_CHECKING:
    import pandas as pd


class AutoARIMADampedModel:
    name = "AutoARIMADamped"
    uses_regressors = False

    def __init__(self, season_length: int = 288, **params: Any) -> None:
        self._season_length = season_length
        phi = params.pop("damping_parameter", None)
        if phi is None:
            phi = params.pop("phi", DEFAULT_DAMPING_PARAMETER)
        else:
            params.pop("phi", None)
        self._phi = float(phi)
        self._params = params

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        from statsforecast import StatsForecast  # noqa: PLC0415
        from statsforecast.models import AutoARIMA  # noqa: PLC0415

        sf = StatsForecast(
            models=[
                AutoARIMA(
                    season_length=self._season_length,
                    alias="AutoARIMADamped",
                    **self._params,
                ),
            ],
            freq=freq,
            n_jobs=1,
        )
        forecast = sf.forecast(df=df, h=horizon, level=levels)
        normalized = _normalize(forecast, model_name="AutoARIMADamped", levels=levels)
        return apply_damping_to_forecast(normalized, phi=self._phi, levels=levels)
