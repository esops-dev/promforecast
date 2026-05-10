"""SeasonalNaive wrapper.

Cheap, robust baseline. Used as the default fallback when an auto-selected
model's backtest MAPE exceeds the configured threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class SeasonalNaiveModel:
    name = "SeasonalNaive"

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
        from statsforecast.models import SeasonalNaive  # noqa: PLC0415

        sf = StatsForecast(
            models=[SeasonalNaive(season_length=self._season_length, **self._params)],
            freq=freq,
            n_jobs=1,
        )
        forecast = sf.forecast(df=df, h=horizon, level=levels)
        return _normalize(forecast, model_name="SeasonalNaive", levels=levels)


def _normalize(forecast: pd.DataFrame, model_name: str, levels: list[int]) -> pd.DataFrame:
    """Map statsforecast's columns to our protocol shape.

    statsforecast emits ``<Model>``, ``<Model>-lo-<level>``, ``<Model>-hi-<level>``;
    we standardise on ``yhat``, ``yhat_lower_<level>``, ``yhat_upper_<level>``.
    """
    rename = {model_name: "yhat"}
    for level in levels:
        rename[f"{model_name}-lo-{level}"] = f"yhat_lower_{level}"
        rename[f"{model_name}-hi-{level}"] = f"yhat_upper_{level}"
    out = forecast.rename(columns=rename).reset_index()
    keep = ["unique_id", "ds", "yhat"] + [
        col for level in levels for col in (f"yhat_lower_{level}", f"yhat_upper_{level}")
    ]
    return out[[c for c in keep if c in out.columns]]
