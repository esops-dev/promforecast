"""Per-group execution and scheduling.

This package replaces what used to be a single 3700+-line ``runner.py``.
The split keeps the public surface — ``Runner``, ``ReloadResult``, and a
handful of test-private helpers — at the package root so existing
``from promforecast.runner import ...`` imports continue to work
unchanged.

Layout:

* :mod:`._scheduler` — the :class:`Runner` orchestration class.
* :mod:`._state` — per-fit / per-group state dataclasses, the
  fit-into-ctx merge helper, and the registry-driven inheritance loop
  for queries skipped on a tick.
* :mod:`._helpers` — pure helpers that don't depend on a live
  ``Runner`` (suffix lookup, backtest summarisation, etc.).
* :mod:`._budgets` — series-budget allocation and per-query overflow /
  canary sampling.
* :mod:`.emission.calibration` — post-fit pure helpers that widen
  forecast bands by per-(model, level) conformal offsets.
* :mod:`.emission.safeguards` — count-aware emission safeguards
  (clip-from-zero, round-to-integer).
* :mod:`.emission.conditional` — pure helpers for conditional forecast
  emission.

Architectural intent: per-family code that does *not* need a live
``Runner`` lives next to other code about the same family; the
``Runner`` class stays in one place because its methods share an
executor, a config, and an exporter, and splitting them across modules
would just trade one long file for a tangle of cross-module references.

Tests monkey-patch ``promforecast.runner.SLOW_FIT_RATIO`` and
``promforecast.runner._summarise_backtest`` directly on this package's
attributes; the scheduler reads both via ``import promforecast.runner
as _pkg`` and ``_pkg.<attr>`` so the patched value is picked up on the
next call. Do not bind those names locally inside the scheduler — that
would silently break the monkey-patch contract.
"""

from __future__ import annotations

# A fit is considered "slow" once its wall-clock crosses this fraction of
# ``safety.fit_timeout``. We pick 0.5 (half) so the warning fires comfortably
# before the timeout actually kills the fit - long enough that operators
# can observe a degraded model before alerts go red, short enough that
# noise-induced 5-10% wobble doesn't trip it.
#
# Defined here (rather than in :mod:`._scheduler`) so test monkey-patches
# of ``promforecast.runner.SLOW_FIT_RATIO`` land on the same attribute the
# scheduler reads via ``_pkg.SLOW_FIT_RATIO``.
SLOW_FIT_RATIO: float = 0.5

# ---- Re-exports -----------------------------------------------------------
# Order matters: ``_scheduler`` does ``import promforecast.runner as _pkg``
# so the package must already exist in ``sys.modules`` (it does, because
# we are mid-``__init__``) and ``SLOW_FIT_RATIO`` above must be set
# before any Runner method runs. The scheduler import is deliberately
# *last* so the package's other re-exports are already in place when the
# scheduler module body executes.

from ._budgets import _apply_canary, _apply_overflow, _compute_global_budgets  # noqa: E402
from ._helpers import (  # noqa: E402
    _FORECAST_METRIC_SUFFIXES,
    _combine_ensemble,
    _format_band_window,
    _pandas_freq,
    _query_id_from_metric,
    _resolve_spec,
    _run_backtest,
    _run_fit,
    _series_key_from_labels,
    _summarise_backtest,
)
from ._scheduler import Runner  # noqa: E402
from ._state import (  # noqa: E402
    _INHERITANCE_REGISTRY,
    ReloadResult,
    _BacktestNeeds,
    _CapacityInput,
    _FitResult,
    _inherit_prior_for_skipped_queries,
    _InheritanceSpec,
    _merge_fit_into_ctx,
    _RunContext,
)
from .emission.calibration import (  # noqa: E402
    _apply_offsets_to_curve_points,
    _apply_offsets_to_snapshot_points,
    _apply_offsets_to_threshold_eta_inputs,
    _calibrated_band_value,
)
from .emission.conditional import _flatten_regressors_to_last_value  # noqa: E402
from .emission.safeguards import (  # noqa: E402
    _ROUND_COUNTER_TRIGGER,
    _apply_safeguards_to_curve_points,
    _apply_safeguards_to_forecast_frame,
    _apply_safeguards_to_snapshot_points,
    _classify_emission_correction,
)

# ``__all__`` covers the test-private helpers as well as the public surface
# so re-exports aren't flagged unused; tests reach into these via
# ``from promforecast.runner import _foo`` patterns. Adding a private here
# is a deliberate choice to keep the test contract stable.
__all__ = [
    "SLOW_FIT_RATIO",
    "_FORECAST_METRIC_SUFFIXES",
    "_INHERITANCE_REGISTRY",
    "_ROUND_COUNTER_TRIGGER",
    "ReloadResult",
    "Runner",
    "_BacktestNeeds",
    "_CapacityInput",
    "_FitResult",
    "_InheritanceSpec",
    "_RunContext",
    "_apply_canary",
    "_apply_offsets_to_curve_points",
    "_apply_offsets_to_snapshot_points",
    "_apply_offsets_to_threshold_eta_inputs",
    "_apply_overflow",
    "_apply_safeguards_to_curve_points",
    "_apply_safeguards_to_forecast_frame",
    "_apply_safeguards_to_snapshot_points",
    "_calibrated_band_value",
    "_classify_emission_correction",
    "_combine_ensemble",
    "_compute_global_budgets",
    "_flatten_regressors_to_last_value",
    "_format_band_window",
    "_inherit_prior_for_skipped_queries",
    "_merge_fit_into_ctx",
    "_pandas_freq",
    "_query_id_from_metric",
    "_resolve_spec",
    "_run_backtest",
    "_run_fit",
    "_series_key_from_labels",
    "_summarise_backtest",
]
