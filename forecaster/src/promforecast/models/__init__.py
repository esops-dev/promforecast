"""Built-in models and the model registry.

External models register via the ``promforecast.models`` entry point group;
see ``pyproject.toml`` for the built-ins.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .auto_arima import AutoARIMAModel
from .auto_ets import AutoETSModel
from .base import ForecastModel
from .mstl import MSTLModel
from .seasonal_naive import SeasonalNaiveModel

# Each factory takes the cascaded ``season_length`` and an arbitrary
# ``params`` dict (per-model overrides from config). ``season_length`` lives
# outside ``params`` because it cascades from ``defaults`` and applies to
# *every* registered model; folding it into ``params`` would force every
# config to repeat it per model.
_BUILTIN: dict[str, Callable[[int, dict[str, Any]], ForecastModel]] = {
    "AutoARIMA": lambda s, p: AutoARIMAModel(season_length=s, **p),
    "AutoETS": lambda s, p: AutoETSModel(season_length=s, **p),
    "MSTL": lambda s, p: MSTLModel(season_length=s, **p),
    "SeasonalNaive": lambda s, p: SeasonalNaiveModel(season_length=s, **p),
}


class UnknownModelError(KeyError):
    """Raised when a configured model name is not registered."""


def build(
    name: str,
    season_length: int,
    params: dict[str, Any] | None = None,
) -> ForecastModel:
    """Instantiate a model by its registered name with optional overrides.

    ``params`` is forwarded as keyword arguments to the model constructor;
    each wrapper accepts ``**params`` and passes them through to the
    underlying ``statsforecast`` model. A typo (e.g. ``stepwize`` vs
    ``stepwise``) therefore surfaces as a ``TypeError`` at the first
    ``fit_predict`` call — not at boot — and is wrapped into the standard
    fit-failure path with the offending model and labels logged.

    A future schema bump can add per-model JSON-schema validation; for now
    we keep the registry shape minimal so external models can register via
    entry points without owning a schema.
    """
    factory = _BUILTIN.get(name)
    if factory is None:
        raise UnknownModelError(name)
    return factory(season_length, dict(params or {}))


__all__ = [
    "AutoARIMAModel",
    "AutoETSModel",
    "ForecastModel",
    "MSTLModel",
    "SeasonalNaiveModel",
    "UnknownModelError",
    "build",
]
