"""MSTL wrapper around statsforecast.

MSTL = Multiple Seasonal-Trend decomposition with LOESS. Decomposes the
series into multiple seasonal components plus trend, then forecasts the
de-seasonalised remainder with a trend model (default: AutoETS) and
re-seasonalises. Useful when a series has more than one seasonality (e.g.
intraday + weekly), but our default config exposes a single ``season_length``
to match the rest of the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .seasonal_naive import _normalize

if TYPE_CHECKING:
    import pandas as pd


class MSTLModel:
    name = "MSTL"
    uses_regressors = False

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
        from statsforecast import StatsForecast  # noqa: PLC0415
        from statsforecast.models import MSTL, AutoETS  # noqa: PLC0415

        sf = StatsForecast(
            models=[
                MSTL(
                    season_length=self._season_length,
                    trend_forecaster=AutoETS(season_length=1),
                    **self._params,
                ),
            ],
            freq=freq,
            n_jobs=1,
        )
        forecast = sf.forecast(df=df, h=horizon, level=levels)
        return _normalize(forecast, model_name="MSTL", levels=levels)
