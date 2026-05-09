"""Built-in models and the model registry.

External models register via the ``promforecast.models`` entry point group;
see ``pyproject.toml`` for the built-ins.
"""

from __future__ import annotations

from collections.abc import Callable

from .auto_arima import AutoARIMAModel
from .auto_ets import AutoETSModel
from .base import ForecastModel
from .seasonal_naive import SeasonalNaiveModel

_BUILTIN: dict[str, Callable[[int], ForecastModel]] = {
    "AutoARIMA": lambda s: AutoARIMAModel(season_length=s),
    "AutoETS": lambda s: AutoETSModel(),
    "SeasonalNaive": lambda s: SeasonalNaiveModel(season_length=s),
}


class UnknownModelError(KeyError):
    """Raised when a configured model name is not registered."""


def build(name: str, season_length: int) -> ForecastModel:
    """Instantiate a model by its registered name."""
    factory = _BUILTIN.get(name)
    if factory is None:
        raise UnknownModelError(name)
    return factory(season_length)


__all__ = ["ForecastModel", "UnknownModelError", "build"]
