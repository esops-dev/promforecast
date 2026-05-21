"""Built-in models and the entry-point-driven model registry.

The registry discovers models through the ``promforecast.models``
entry-point group. Built-in wrappers (``AutoARIMA``, ``AutoETS``, ``MSTL``,
``SeasonalNaive``) are registered by this package's own ``pyproject.toml``;
third-party packages (such as ``promforecast-prophet``) register additional
models the same way and they appear in the registry without any forecaster-
side code changes.

Each entry point resolves to a class implementing the
:class:`ForecastModel` protocol. The class is instantiated as
``Cls(season_length=<int>, **params)`` per :func:`build`. ``season_length``
is passed as a kwarg because it cascades from ``defaults`` and applies to
every model — folding it into ``params`` would force every config to repeat
it per model. ``params`` is forwarded verbatim so the wrapper (and through
it, ``statsforecast`` or any third-party library) sees its tunables.

Discovery is cached so the entry-point scan happens once per process. The
cache exposes :func:`refresh` for tests and the future ``/-/reload`` model
hot-reload story; production code never needs to call it.
"""

from __future__ import annotations

import threading
from importlib.metadata import entry_points
from typing import Any

import structlog

from .auto_arima import AutoARIMAModel
from .auto_arima_damped import AutoARIMADampedModel
from .auto_ets import AutoETSModel
from .auto_ets_damped import AutoETSDampedModel
from .base import ForecastModel
from .mstl import MSTLModel
from .seasonal_naive import SeasonalNaiveModel

logger = structlog.get_logger(__name__)

ENTRY_POINT_GROUP = "promforecast.models"


class UnknownModelError(KeyError):
    """Raised when a configured model name is not registered."""


class ModelRegistrationError(RuntimeError):
    """Raised when an entry point can't be loaded into a usable class.

    A broken third-party package (import error, missing class, wrong type)
    must not silently disappear from the registry — operators would see a
    confusing ``UnknownModelError`` instead of the real cause. This error
    is logged at registry-build time with the entry-point name and the
    underlying exception so the offending package is unambiguous.
    """


_lock = threading.Lock()
_cache: dict[str, type[ForecastModel]] | None = None


def _discover() -> dict[str, type[ForecastModel]]:
    """Scan ``promforecast.models`` entry points and return ``{name: class}``.

    The display name (e.g. ``AutoARIMA``) comes from the registered class's
    ``name`` attribute, *not* the entry-point name. This keeps configs
    reading naturally (``- AutoARIMA``) and lets entry-point names follow
    PEP 503 case-folding without affecting the public surface.

    Duplicate model names raise :class:`ModelRegistrationError` so a
    third-party package can't silently shadow a built-in (or another
    plug-in). Broken entry points are logged and skipped — one bad plug-in
    must not take down the whole forecaster.
    """
    discovered: dict[str, type[ForecastModel]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
        except Exception as exc:  # pragma: no cover - exercised via tests
            logger.error(
                "model_entry_point_load_failed",
                entry_point=ep.name,
                value=ep.value,
                error=str(exc),
            )
            continue
        if not isinstance(obj, type):
            logger.error(
                "model_entry_point_not_a_class",
                entry_point=ep.name,
                value=ep.value,
                resolved_type=type(obj).__name__,
            )
            continue
        name = getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            logger.error(
                "model_entry_point_missing_name",
                entry_point=ep.name,
                value=ep.value,
            )
            continue
        if name in discovered and discovered[name] is not obj:
            raise ModelRegistrationError(
                f"duplicate model name {name!r}: "
                f"{discovered[name].__module__}.{discovered[name].__qualname__} "
                f"and {obj.__module__}.{obj.__qualname__} both registered"
            )
        discovered[name] = obj
    return discovered


def _registry() -> dict[str, type[ForecastModel]]:
    """Lazy, thread-safe accessor for the discovered registry.

    The first caller pays the entry-point scan cost (~milliseconds);
    subsequent callers get the cached map. The lock guards against two
    concurrent runs racing during a hot-reload that calls :func:`refresh`.
    """
    global _cache  # noqa: PLW0603 — module-level cache is the whole point
    cached = _cache
    if cached is not None:
        return cached
    with _lock:
        if _cache is None:
            _cache = _discover()
        return _cache


def refresh() -> None:
    """Drop the cached registry so the next call re-scans entry points.

    Useful for tests that monkeypatch ``importlib.metadata.entry_points``,
    and for any future model-set hot-reload that shouldn't require a pod
    restart. Cheap — the next :func:`build` rebuilds on demand.
    """
    global _cache  # noqa: PLW0603 — module-level cache is the whole point
    with _lock:
        _cache = None


def available() -> list[str]:
    """Return the sorted list of registered model names.

    Surfaces the live registry to operators (validate CLI, /healthz future
    extensions, error messages) without leaking the concrete class types.
    """
    return sorted(_registry().keys())


def build(
    name: str,
    season_length: int,
    params: dict[str, Any] | None = None,
) -> ForecastModel:
    """Instantiate a registered model with optional per-model overrides.

    ``params`` is forwarded as keyword arguments to the model's
    constructor; each wrapper accepts ``**params`` and passes them through
    to its underlying library. A typo (e.g. ``stepwize`` vs ``stepwise``)
    therefore surfaces as a ``TypeError`` at the first ``fit_predict``
    call — not at boot — and is wrapped into the standard fit-failure
    path with the offending model and labels logged.

    ``UnknownModelError`` carries the available model list in its message
    so a misspelled name surfaces the alternatives without hunting through
    docs.
    """
    registry = _registry()
    cls = registry.get(name)
    if cls is None:
        raise UnknownModelError(f"unknown model {name!r}; available: {', '.join(sorted(registry))}")
    # The ``ForecastModel`` protocol intentionally describes only the
    # runtime ``fit_predict`` shape, not constructor parameters — every
    # built-in (and every plug-in) takes ``season_length`` plus ``**params``
    # by registry contract. mypy can't infer that from the Protocol alone.
    return cls(season_length=season_length, **(params or {}))  # type: ignore[call-arg]


__all__ = [
    "ENTRY_POINT_GROUP",
    "AutoARIMADampedModel",
    "AutoARIMAModel",
    "AutoETSDampedModel",
    "AutoETSModel",
    "ForecastModel",
    "MSTLModel",
    "ModelRegistrationError",
    "SeasonalNaiveModel",
    "UnknownModelError",
    "available",
    "build",
    "refresh",
]
