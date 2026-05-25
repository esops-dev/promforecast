"""Change-correlation explanation for ``forecast_drift_score`` events.

When ``forecast_drift_score`` crosses its configured threshold, the
runner emits a structured ``forecast_drift_explained`` log event whose
fields collect "what else moved at the same time?" context. That turns
a generic drift alert ("something drifted") into an actionable one
("start your investigation on the ``deploys`` regressor — it explains
0.82 of the drift variance over the last 30 fits").

Two suspect families coexist on every explanation:

* **``top_value_suspect``** — the regressor with the largest absolute
  value at the drift moment. Useful when the drift is driven by a
  one-shot event (a deploy that just fired); the regressor is "the
  thing that's loudest right now".
* **``top_correlation_suspect``** — the regressor whose value history
  most predicts the drift_score history over the rolling
  ``correlation_window``. Useful when the drift is driven by a slower
  signal that's been trending alongside the forecast.

The ``primary_suspect`` label on the
``forecast_drift_explained_total`` counter prefers the
correlation-based suspect when the rolling window has enough samples
for a meaningful Pearson coefficient; it falls back to the
value-based suspect when correlation is ``insufficient_data``.

Pure helpers — the runner owns the rolling-window state and the log
emission; this module computes the suspects and the correlation
math.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

CorrelationConfidence = Literal["high", "medium", "low", "insufficient_data"]

# Minimum sample size to compute a Pearson correlation at all. Below
# this, the explanation falls back to the value-based suspect.
_MIN_PAIRS_FOR_CORRELATION = 4
# Confidence thresholds tuned for "roughly aligned with operator
# intuition": at high confidence the operator should investigate the
# suspect first; at medium they should double-check; at low the
# correlation exists but the relationship is weak.
_CONFIDENCE_HIGH_ABS = 0.7
_CONFIDENCE_HIGH_N = 20
_CONFIDENCE_MEDIUM_ABS = 0.4
_CONFIDENCE_MEDIUM_N = 8


@dataclass(frozen=True)
class _Sample:
    """One historical observation for the rolling window."""

    drift_score: float
    regressor_values: dict[str, float]


@dataclass
class DriftWindow:
    """Per-(group, query) ring buffer of drift + regressor observations.

    The maxlen is bounded by the operator's ``correlation_window``
    config so a long-running deployment doesn't accumulate unbounded
    history. ``record`` appends; ``rebuild`` is used when the operator
    reloads with a different window size (so the buffer respects the
    new bound on the next fit).
    """

    maxlen: int
    _samples: deque[_Sample] = field(init=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=max(1, self.maxlen))

    def record(self, drift_score: float, regressor_values: dict[str, float]) -> None:
        """Append one observation, evicting the oldest when full."""
        self._samples.append(
            _Sample(drift_score=drift_score, regressor_values=dict(regressor_values))
        )

    def samples(self) -> list[_Sample]:
        """Return a snapshot of the current window (oldest-first)."""
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


@dataclass(frozen=True)
class DriftExplanation:
    """Composite explanation for one drift threshold-crossing event.

    ``primary_suspect`` is the operator-facing label on the
    ``forecast_drift_explained_total`` counter — preferring the
    correlation-based suspect when correlation confidence is high or
    medium, falling back to the value-based suspect otherwise (and
    to ``"unknown"`` when neither candidate produced a finite answer).
    """

    primary_suspect: str
    top_value_suspect: str
    top_regressor_correlation: float
    top_regressor_correlation_confidence: CorrelationConfidence
    top_correlation_suspect: str
    correlation_window_size: int


def compute_pearson(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Return ``(coefficient, n_finite_pairs)``.

    ``n_finite_pairs`` is the count of pairs where both ``x`` and
    ``y`` are finite — non-finite values are skipped so a single NaN
    regressor reading doesn't poison the coefficient. When fewer than
    :data:`_MIN_PAIRS_FOR_CORRELATION` finite pairs exist, the
    coefficient is reported as ``NaN`` (the caller treats this as
    ``insufficient_data``).

    Division-by-zero (zero variance on either side) returns ``NaN``
    rather than raising — a flat regressor over the window can't
    explain anything, and the caller should fall back to the
    value-based suspect.
    """
    if len(xs) != len(ys):
        raise ValueError("compute_pearson: xs and ys must have equal length")
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < _MIN_PAIRS_FOR_CORRELATION:
        return float("nan"), n
    sum_x = sum(p[0] for p in pairs)
    sum_y = sum(p[1] for p in pairs)
    mean_x = sum_x / n
    mean_y = sum_y / n
    cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
    var_x = sum((p[0] - mean_x) ** 2 for p in pairs)
    var_y = sum((p[1] - mean_y) ** 2 for p in pairs)
    if var_x == 0 or var_y == 0:
        return float("nan"), n
    return cov / math.sqrt(var_x * var_y), n


def classify_confidence(coefficient: float, n_pairs: int) -> CorrelationConfidence:
    """Map ``(|coefficient|, n_pairs)`` to one of four named bands.

    The bands are deliberately coarse — operators read them as "should
    I trust this suspect?", not as a p-value. The cutoffs are sized
    around what an operator would call "really worth investigating
    first": ``high`` requires both a strong correlation *and* enough
    samples, ``medium`` accepts one of the two.
    """
    if not math.isfinite(coefficient) or n_pairs < _MIN_PAIRS_FOR_CORRELATION:
        return "insufficient_data"
    abs_coef = abs(coefficient)
    if abs_coef >= _CONFIDENCE_HIGH_ABS and n_pairs >= _CONFIDENCE_HIGH_N:
        return "high"
    if abs_coef >= _CONFIDENCE_MEDIUM_ABS and n_pairs >= _CONFIDENCE_MEDIUM_N:
        return "medium"
    return "low"


def explain(
    window: DriftWindow,
    current_regressor_values: dict[str, float],
) -> DriftExplanation:
    """Produce a :class:`DriftExplanation` from a rolling window snapshot.

    The current observation is *also* fed in (separate from the
    window) so the value-based suspect always reflects "what was loud
    at the drift moment" — even when the window itself is empty
    (e.g. a fresh install whose very first fit already trips the
    drift threshold).

    Returns a sentinel explanation (``primary_suspect="unknown"``,
    ``correlation_confidence="insufficient_data"``) when no regressors
    are configured / available; the caller still emits the structured
    log line so the operator can see "drift fired but no suspects" as
    its own signal.
    """
    samples = window.samples()
    n_samples = len(samples)
    top_value_suspect = _select_top_value_suspect(current_regressor_values)
    top_correlation_suspect, coefficient = _select_top_correlation_suspect(samples)
    confidence = classify_confidence(coefficient, n_samples)
    # Primary suspect picks correlation when the band is high/medium —
    # those are "trust this candidate enough to start the
    # investigation". Low / insufficient_data falls back to the
    # value-based suspect, which never has confidence semantics (it's
    # a snapshot of "what's loud right now").
    primary = top_correlation_suspect if confidence in ("high", "medium") else top_value_suspect
    if not primary:
        primary = "unknown"
    return DriftExplanation(
        primary_suspect=primary,
        top_value_suspect=top_value_suspect or "unknown",
        top_regressor_correlation=coefficient,
        top_regressor_correlation_confidence=confidence,
        top_correlation_suspect=top_correlation_suspect or "unknown",
        correlation_window_size=n_samples,
    )


def _select_top_value_suspect(regressor_values: dict[str, float]) -> str:
    """Pick the regressor with the largest absolute value (or empty string)."""
    best_id: str = ""
    best_abs: float = float("-inf")
    for regressor_id, value in regressor_values.items():
        if not math.isfinite(value):
            continue
        abs_val = abs(float(value))
        if abs_val > best_abs:
            best_abs = abs_val
            best_id = regressor_id
    return best_id


def _select_top_correlation_suspect(samples: list[_Sample]) -> tuple[str, float]:
    """Walk the window and return the (regressor_id, coefficient) with max |corr|.

    Returns ``("", NaN)`` when the window is too short for any
    correlation. Ties on absolute correlation break by regressor id
    alphabetically so successive runs report a stable suspect when
    several regressors share the same coefficient (e.g. a uniform
    flat signal across all of them).
    """
    if len(samples) < _MIN_PAIRS_FOR_CORRELATION:
        return "", float("nan")
    # Pivot the per-sample dict-of-values into per-regressor lists,
    # filling missing samples with NaN so the per-pair filter in
    # ``compute_pearson`` drops them cleanly.
    regressor_ids: set[str] = set()
    for sample in samples:
        regressor_ids.update(sample.regressor_values)
    if not regressor_ids:
        return "", float("nan")
    drift_series = [sample.drift_score for sample in samples]
    best_id: str = ""
    best_abs: float = float("-inf")
    best_signed: float = float("nan")
    for regressor_id in sorted(regressor_ids):
        reg_series = [sample.regressor_values.get(regressor_id, float("nan")) for sample in samples]
        coef, _ = compute_pearson(drift_series, reg_series)
        if not math.isfinite(coef):
            continue
        abs_coef = abs(coef)
        if abs_coef > best_abs:
            best_abs = abs_coef
            best_id = regressor_id
            best_signed = coef
    if not best_id:
        return "", float("nan")
    return best_id, best_signed
