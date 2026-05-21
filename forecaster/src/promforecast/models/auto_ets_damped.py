"""AutoETSDamped wrapper around statsforecast.

Variant of :class:`AutoETSModel` that forces a *damped* additive trend.
Unbounded linear extrapolation is exactly what bites capacity forecasts at
multi-day horizons — a disk that grew 1%/day for a week does not keep
growing 1%/day forever. Damping geometrically decays the per-step trend
contribution so the forecast bends toward a level instead of running to
infinity.

The single tunable ``damping_parameter`` (alias: ``phi``) is forwarded
to statsforecast's native ``phi`` argument and into a post-fit damping
pass applied to the predicted values and bands. The post-fit pass keeps
the damping behaviour consistent across (model, params) combinations
where the statsforecast auto-selector might otherwise pick a non-damped
candidate — the user opted into damping by choosing this wrapper, so the
damping is enforced rather than left to the search.

A value of 1.0 disables damping (and the model reduces to vanilla
``AutoETS`` with a forced additive trend); values below 1.0 bend the
forecast curve, with smaller values producing tighter bends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .damping import DEFAULT_DAMPING_PARAMETER, apply_damping_to_forecast
from .seasonal_naive import _normalize

if TYPE_CHECKING:
    import pandas as pd


class AutoETSDampedModel:
    name = "AutoETSDamped"

    def __init__(self, season_length: int = 288, **params: Any) -> None:
        self._season_length = season_length
        # Accept both ``damping_parameter`` (preferred, matches the
        # roadmap-documented config knob) and ``phi`` (matches the
        # underlying statsforecast argument). The two are aliases; if both
        # are set, ``damping_parameter`` wins because that's the name in
        # the config example we documented.
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
        from statsforecast.models import AutoETS  # noqa: PLC0415

        sf = StatsForecast(
            models=[
                AutoETS(
                    season_length=self._season_length,
                    damped=True,
                    phi=self._phi,
                    alias="AutoETSDamped",
                    **self._params,
                ),
            ],
            freq=freq,
            n_jobs=1,
        )
        forecast = sf.forecast(df=df, h=horizon, level=levels)
        normalized = _normalize(forecast, model_name="AutoETSDamped", levels=levels)
        return apply_damping_to_forecast(normalized, phi=self._phi, levels=levels)
