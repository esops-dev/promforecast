"""Pure helpers for conditional forecast emission.

The full conditional refit / calibration orchestration lives on
``Runner`` because it needs the per-instance fit executor and exporter
references; this module collects the pure-function helpers shared with
the runner so the data manipulation pieces can be reasoned about in
isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _flatten_regressors_to_last_value(
    regressors: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Return a copy of ``regressors`` whose every row equals the last row.

    Implements the "held at current value" semantics of the conditional
    forecast: each regressor column is replaced with a constant series
    equal to its most recent in-sample value. Calendar features (which
    legitimately vary per timestamp) are preserved through the same
    flattening — the conditional contract is "imagine current state
    continued unchanged", which by construction freezes the calendar as
    well. Returns ``None`` when no regressors were configured so the
    caller skips the re-fit's regressor argument cleanly.

    Uses scalar broadcast (``flat[col] = scalar``) instead of building a
    Python list per column so a wide regressor frame doesn't materialise
    N copies of the same value through the heap. pandas treats the
    scalar assignment as a constant broadcast and writes a single typed
    array.
    """
    if regressors is None or regressors.empty:
        return regressors
    import pandas as pd  # noqa: PLC0415

    last_row = regressors.iloc[-1]
    flat = pd.DataFrame(index=regressors.index)
    for col in regressors.columns:
        if col == "ds":
            flat[col] = regressors[col].to_numpy()
            continue
        # Scalar broadcast: pandas writes a constant array of the right
        # dtype without allocating a Python list of N references.
        flat[col] = last_row[col]
    return flat
