"""Forecast-model protocol.

Implementations live alongside this file (``auto_arima.py`` etc.) and are
discoverable via the ``promforecast.models`` entry-point group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class ForecastModel(Protocol):
    """Pluggable forecast model.

    Implementations must be safe to instantiate without arguments and must
    not mutate inputs.
    """

    name: str

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fit and forecast in one shot.

        Args:
            df: Historical series with columns ``unique_id``, ``ds``, ``y``.
            horizon: Number of future steps to predict.
            freq: Pandas-style frequency string (e.g. ``"5min"``).
            levels: Confidence levels in percent (e.g. ``[80, 95]``).
            regressors: Optional exogenous regressors aligned to ``df``.

        Returns:
            DataFrame with columns ``unique_id``, ``ds``, ``yhat``,
            ``yhat_lower_<level>``, ``yhat_upper_<level>``.
        """
        ...
