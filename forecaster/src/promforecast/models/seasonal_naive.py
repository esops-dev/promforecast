"""SeasonalNaive wrapper.

Cheap, robust baseline. Used as the default fallback when an auto-selected
model's backtest MAPE exceeds the configured threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class SeasonalNaiveModel:
    name = "SeasonalNaive"

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
