"""AutoETS wrapper around statsforecast.

Exponential-smoothing state-space model with automatic component selection
(error/trend/seasonality). Cheaper than AutoARIMA on series with strong
seasonality and clean noise; struggles on bursty or zero-inflated series
where AutoARIMA's differencing wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .seasonal_naive import _normalize

if TYPE_CHECKING:
    import pandas as pd


class AutoETSModel:
    name = "AutoETS"

    def __init__(self, season_length: int = 288) -> None:
        self._season_length = season_length

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        from statsforecast import StatsForecast  # noqa: PLC0415
        from statsforecast.models import AutoETS  # noqa: PLC0415

        sf = StatsForecast(
            models=[AutoETS(season_length=self._season_length)],
            freq=freq,
            n_jobs=1,
        )
        forecast = sf.forecast(df=df, h=horizon, level=levels)
        return _normalize(forecast, model_name="AutoETS", levels=levels)
