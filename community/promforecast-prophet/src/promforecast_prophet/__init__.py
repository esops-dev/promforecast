"""Prophet forecasting model plug-in for promforecast.

Importing this package is cheap — the heavy ``prophet`` import is deferred
until the first ``fit_predict`` call so the registry's entry-point scan
stays fast even on systems where Stan is slow to import.
"""

from __future__ import annotations

from .model import ProphetModel

__all__ = ["ProphetModel"]
