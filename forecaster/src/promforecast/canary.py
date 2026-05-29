"""Canary-vs-baseline forecast comparison.

Splits a query's fetched series into a *baseline* population (stable pods)
and a *canary* population (the new version) using Prometheus-style label
matchers, builds an aggregate baseline series, and lets the runner project
what the canary *should* look like. Each canary series' most recent actual
is then compared to the baseline-predicted forecast band, emitting
``<id>_canary_forecast_deviation`` in the same shape as the existing
deviation metrics.

Only ``=`` and ``!=`` matchers are supported: the selectors partition an
already-fetched result set rather than being pushed down to the TSDB. That
keeps the feature decoupled from PromQL parsing and matches how Argo
Rollouts / Flagger express their canary vs stable label filters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source import SeriesFrame


class CanarySelectorError(ValueError):
    """Raised when a canary selector cannot be parsed."""


@dataclass(frozen=True)
class LabelMatcher:
    key: str
    op: str  # "=" or "!="
    value: str

    def matches(self, labels: dict[str, str]) -> bool:
        actual = labels.get(self.key, "")
        if self.op == "=":
            return actual == self.value
        return actual != self.value


# ``key="value"`` or ``key!="value"`` — double-quoted value, the only form
# Prometheus selectors use for equality matchers.
_MATCHER_RE = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=)\s*"((?:[^"\\]|\\.)*)"\s*$')


def parse_selector(selector: str) -> list[LabelMatcher]:
    """Parse ``{k="v", k!="v"}`` into a list of :class:`LabelMatcher`.

    The surrounding braces are optional. An empty selector matches
    everything (returns no matchers). Raises :class:`CanarySelectorError`
    on an unsupported matcher (regex ``=~`` / ``!~`` or malformed token).
    """
    text = selector.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    text = text.strip()
    if not text:
        return []
    matchers: list[LabelMatcher] = []
    for part in _split_top_level(text):
        token = part.strip()
        if not token:
            continue
        if "=~" in token or "!~" in token:
            raise CanarySelectorError(
                f"regex matchers are not supported in canary selectors: {token!r}"
            )
        m = _MATCHER_RE.match(token)
        if m is None:
            raise CanarySelectorError(f"unparseable canary selector matcher: {token!r}")
        key, op, value = m.group(1), m.group(2), m.group(3)
        matchers.append(LabelMatcher(key=key, op=op, value=value.replace('\\"', '"')))
    return matchers


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside a quoted value."""
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
            continue
        if ch == "," and not in_quote:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def matches_all(labels: dict[str, str], matchers: list[LabelMatcher]) -> bool:
    return all(m.matches(labels) for m in matchers)


def split_frames(
    frames: list[SeriesFrame],
    *,
    canary_selector: str,
    baseline_selector: str,
) -> tuple[list[SeriesFrame], list[SeriesFrame]]:
    """Partition ``frames`` into (canary, baseline) by their label selectors.

    A series may match neither selector (dropped) but not both — overlapping
    selectors are a config error, surfaced as :class:`CanarySelectorError`.
    """
    canary_matchers = parse_selector(canary_selector)
    baseline_matchers = parse_selector(baseline_selector)
    canary: list[SeriesFrame] = []
    baseline: list[SeriesFrame] = []
    for frame in frames:
        in_canary = matches_all(frame.labels, canary_matchers)
        in_baseline = matches_all(frame.labels, baseline_matchers)
        if in_canary and in_baseline:
            raise CanarySelectorError(
                f"series {frame.labels!r} matches both canary and baseline selectors; "
                "the selectors must be mutually exclusive"
            )
        if in_canary:
            canary.append(frame)
        elif in_baseline:
            baseline.append(frame)
    return canary, baseline


def aggregate_baseline(frames: list[SeriesFrame]) -> tuple[list[datetime], list[float]]:
    """Mean-across-baseline-series, grouped + sorted by timestamp.

    Returns ``([], [])`` when no usable points exist so the caller can skip
    the fit gracefully.
    """
    sums: dict[datetime, float] = {}
    counts: dict[datetime, int] = {}
    for frame in frames:
        for ts, val in zip(frame.timestamps, frame.values, strict=True):
            sums[ts] = sums.get(ts, 0.0) + float(val)
            counts[ts] = counts.get(ts, 0) + 1
    if not sums:
        return [], []
    ordered = sorted(sums)
    values = [sums[ts] / counts[ts] for ts in ordered]
    return ordered, values


def latest_actual(frame: SeriesFrame) -> float | None:
    """The most recent (by timestamp) value of a canary series, or ``None``."""
    if not frame.values:
        return None
    pairs = sorted(zip(frame.timestamps, frame.values, strict=True), key=lambda p: p[0])
    return float(pairs[-1][1])
