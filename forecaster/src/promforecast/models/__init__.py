"""Built-in models and the model registry.

External models register via the ``promforecast.models`` entry point group;
see ``pyproject.toml`` for the built-ins.
"""

from .base import ForecastModel

__all__ = ["ForecastModel"]
