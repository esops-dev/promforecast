"""AutoARIMA wrapper around statsforecast."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .seasonal_naive import _normalize

if TYPE_CHECKING:
    import pandas as pd


class AutoARIMAModel:
    name = "AutoARIMA"
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
        from statsforecast.models import AutoARIMA  # noqa: PLC0415

        sf = StatsForecast(
            models=[AutoARIMA(season_length=self._season_length, **self._params)],
            freq=freq,
            n_jobs=1,
        )
        forecast = sf.forecast(df=df, h=horizon, level=levels)
        return _normalize(forecast, model_name="AutoARIMA", levels=levels)
