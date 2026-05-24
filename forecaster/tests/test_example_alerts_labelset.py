"""Audit ``examples/alerts/promforecast-rules.yaml`` against the sink+VM
labelset that the sink-first defaults emit.

The forecaster ships two write paths for forecast metrics: the
``/metrics`` snapshot scraped by Prometheus (carries the forecaster's own
``instance="<pod>:9091"`` and ``job="<release>"``) and the
``sink.remote_write`` push into a long-term TSDB (carries only the source
metric's labels — no forecaster scrape labels). With the umbrella chart
defaulting to sink-on and the standalone chart's
``serviceMonitor.dropForecastSnapshots`` default deriving from
``sink.remote_write.enabled``, the bundled alerts should only filter on
source-side labels. A literal ``instance="<pod>:9091"`` matcher would
silently match zero series under the recommended path.

This test:

* parses every alert's ``expr``,
* extracts every leaf metric reference and its label matchers,
* asserts no rule filters a ``*_forecast`` family metric on a label that
  only the dropped snapshot copy would carry, and
* asserts every referenced metric and matcher resolves to at least one
  series in a synthetic sink-shaped labelset fixture.

It is deliberately a structural audit rather than a full PromQL
evaluator — the goal is to fail loudly when an example rule drifts away
from the labelset the sink actually emits, which is the failure mode
operators hit when they paste a stale example into a real cluster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPO_ROOT / "examples" / "alerts" / "promforecast-rules.yaml"

# ---------------------------------------------------------------------------
# Sink-shaped labelset fixture.
#
# Models a small but representative slice of what the sink+VM path emits
# after a single fit on the bundled dev stack. ``*_forecast`` /
# ``*_forecast_lower`` / ``*_forecast_upper`` carry source labels only
# (``instance`` from node-exporter, ``device`` / ``mountpoint`` for disk
# series). Operational metrics emitted only on /metrics (``forecast_*``,
# ``*_forecast_time_to_threshold_seconds``, ``*_forecast_growth_rate``)
# carry both the source labels (where applicable) and the forecaster's
# own scrape labels (``job="promforecast"``, ``instance="forecaster:9091"``).
# A rule that filters on labels the sink does not project will return
# zero series under the recommended path; this fixture catches that drift
# at PR time.
# ---------------------------------------------------------------------------

_SINK_FORECAST_LABELS = [
    {
        "id": "node_cpu_busy_pct",
        "group": "node_capacity",
        "model": "AutoARIMA",
        "horizon": "next",
        "instance": "node-exporter:9100",
    },
    {
        "id": "node_memory_used_bytes",
        "group": "node_capacity",
        "model": "SeasonalNaive",
        "horizon": "next",
        "instance": "node-exporter:9100",
    },
    {
        "id": "node_filesystem_avail_bytes",
        "group": "node_storage",
        "model": "AutoARIMA",
        "horizon": "next",
        "instance": "node-exporter:9100",
        "mountpoint": "/var",
        "device": "sda1",
        "fstype": "ext4",
    },
]

_SNAPSHOT_OP_LABELS = {
    "instance": "forecaster:9091",
    "job": "promforecast",
}


def _with(labels: dict[str, str], **extra: str) -> dict[str, str]:
    out = dict(labels)
    out.update(extra)
    return out


def _expand_levels(base: list[dict[str, str]], levels: tuple[str, ...]) -> list[dict[str, str]]:
    """Replicate a base labelset across the configured confidence levels."""
    return [_with(b, level=lvl) for b in base for lvl in levels]


LEVELS = ("80", "95")
THRESHOLD_NAMES = ("warn", "critical")
COMPARATORS = ("lt", "gt")

FIXTURE: dict[str, list[dict[str, str]]] = {
    # Sink-emitted forecast curves: source labels only, no forecaster
    # scrape labels.
    "node_cpu_busy_pct_forecast": _SINK_FORECAST_LABELS[:1] + _SINK_FORECAST_LABELS[1:2],
    "node_memory_used_bytes_forecast": _SINK_FORECAST_LABELS[1:2],
    "node_filesystem_avail_bytes_forecast": _SINK_FORECAST_LABELS[2:3],
    # Bands at every configured level.
    "node_cpu_busy_pct_forecast_lower": _expand_levels(_SINK_FORECAST_LABELS[:2], LEVELS),
    "node_cpu_busy_pct_forecast_upper": _expand_levels(_SINK_FORECAST_LABELS[:2], LEVELS),
    "node_memory_used_bytes_forecast_lower": _expand_levels(_SINK_FORECAST_LABELS[1:2], LEVELS),
    "node_memory_used_bytes_forecast_upper": _expand_levels(_SINK_FORECAST_LABELS[1:2], LEVELS),
    "node_filesystem_avail_bytes_forecast_lower": _expand_levels(
        _SINK_FORECAST_LABELS[2:3], LEVELS
    ),
    "node_filesystem_avail_bytes_forecast_upper": _expand_levels(
        _SINK_FORECAST_LABELS[2:3], LEVELS
    ),
    # Snapshot-only operational metrics carry forecaster scrape labels.
    "forecast_last_run_timestamp_seconds": [
        _with(_SNAPSHOT_OP_LABELS, group="node_capacity"),
        _with(_SNAPSHOT_OP_LABELS, group="node_storage"),
    ],
    "forecast_accuracy_mape": [
        _with(
            _SNAPSHOT_OP_LABELS,
            group="node_capacity",
            id="node_cpu_busy_pct",
            model="AutoARIMA",
            horizon="24h",
        ),
        _with(
            _SNAPSHOT_OP_LABELS,
            group="node_storage",
            id="node_filesystem_avail_bytes",
            model="AutoARIMA",
            horizon="24h",
        ),
    ],
    "forecast_accuracy_mase": [
        _with(
            _SNAPSHOT_OP_LABELS,
            group="node_capacity",
            id="node_cpu_busy_pct",
            model="AutoARIMA",
            horizon="24h",
        ),
    ],
    "forecast_quality_score": [
        _with(_SNAPSHOT_OP_LABELS, group=g, id=i, model=m, model_fallback=mf, cold_start="false")
        for g, i, m in [
            ("node_capacity", "node_cpu_busy_pct", "AutoARIMA"),
            ("node_capacity", "node_memory_used_bytes", "SeasonalNaive"),
            ("node_storage", "node_filesystem_avail_bytes", "AutoARIMA"),
        ]
        # Both fallback states present so ``ForecastFallbackPersistent``
        # (filters on ``model_fallback="true"``) and the cold-start /
        # quality alerts (filter on ``cold_start!="true"``) both resolve.
        for mf in ("false", "true")
    ],
    "forecast_deviation_outside_band": [
        _with(_SNAPSHOT_OP_LABELS, group=g, id=i, model=m, level=lvl)
        for g, i, m in [
            ("node_capacity", "node_cpu_busy_pct", "AutoARIMA"),
            ("node_storage", "node_filesystem_avail_bytes", "AutoARIMA"),
        ]
        for lvl in LEVELS
    ],
    "forecast_deviation_ratio": [
        _with(_SNAPSHOT_OP_LABELS, group=g, id=i, model=m, level=lvl)
        for g, i, m in [
            ("node_capacity", "node_cpu_busy_pct", "AutoARIMA"),
            ("node_storage", "node_filesystem_avail_bytes", "AutoARIMA"),
        ]
        for lvl in LEVELS
    ],
    "forecast_cold_start_siblings": [
        _with(_SNAPSHOT_OP_LABELS, group="node_capacity", query="node_cpu_busy_pct"),
    ],
    "node_filesystem_avail_bytes_forecast_time_to_threshold_seconds": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="node_filesystem_avail_bytes",
            group="node_storage",
            model="AutoARIMA",
            instance="node-exporter:9100",
            mountpoint="/var",
            name=name,
            comparator=cmp,
            level=lvl,
        )
        for name in THRESHOLD_NAMES
        for cmp in COMPARATORS
        for lvl in LEVELS
    ],
    "node_cpu_busy_pct_forecast_time_to_threshold_seconds": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="node_cpu_busy_pct",
            group="node_capacity",
            model="AutoARIMA",
            instance="node-exporter:9100",
            name=name,
            comparator=cmp,
            level=lvl,
        )
        for name in THRESHOLD_NAMES
        for cmp in COMPARATORS
        for lvl in LEVELS
    ],
    "node_filesystem_avail_bytes_forecast_growth_rate": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="node_filesystem_avail_bytes",
            group="node_storage",
            model="AutoARIMA",
            instance="node-exporter:9100",
            mountpoint="/var",
            window=w,
        )
        for w in ("1d", "7d")
    ],
    "node_filesystem_avail_bytes_forecast_will_breach": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="node_filesystem_avail_bytes",
            group="node_storage",
            model="AutoARIMA",
            instance="node-exporter:9100",
            mountpoint="/var",
            name=name,
            severity=sev,
            level=lvl,
        )
        for name in THRESHOLD_NAMES
        for sev in ("warning", "critical")
        for lvl in LEVELS
    ],
    "kafka_consumer_lag_forecast_time_to_drain_seconds": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="kafka_consumer_lag",
            group="kafka",
            model="AutoARIMA",
            topic="orders",
            name="drained",
        ),
    ],
    "http_request_error_ratio_forecast_error_budget_exhausted_in_seconds": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="http_request_error_ratio",
            group="http_slo",
            model="AutoARIMA",
            service="checkout",
            window="30d",
        ),
    ],
    "http_request_error_ratio_forecast_error_budget_burn_rate": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="http_request_error_ratio",
            group="http_slo",
            model="AutoARIMA",
            service="checkout",
            window="30d",
        ),
    ],
    "http_request_error_ratio_forecast_error_budget_remaining_seconds": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="http_request_error_ratio",
            group="http_slo",
            model="AutoARIMA",
            service="checkout",
            window="30d",
        ),
    ],
    "forecast_recommended_maintenance_window": [
        _with(
            _SNAPSHOT_OP_LABELS,
            id="node_cpu_busy_pct",
            group="node_capacity",
            model="AutoARIMA",
            instance="node-exporter:9100",
            rank=str(r),
            duration_seconds="1800",
            start_timestamp_seconds="1700000000",
        )
        for r in (1, 2, 3)
    ],
    "forecast_deploy_risk_score": [
        _with(
            _SNAPSHOT_OP_LABELS,
            service="payments",
            deployment_id="payments-1715000000",
            contributing_signals="2",
        ),
        _with(
            _SNAPSHOT_OP_LABELS,
            service="search",
            deployment_id="search-1715000100",
            contributing_signals="3",
        ),
    ],
    "forecast_incident_recurrence_score": [
        _with(
            _SNAPSHOT_OP_LABELS,
            group="incidents",
            id="pagerduty_incidents_total",
            day_of_week=str(d),
            hour_of_day=str(h),
        )
        for d in (0, 1, 2)
        for h in (0, 12, 23)
    ],
}

# ---------------------------------------------------------------------------
# PromQL leaf-matcher extraction.
#
# We only need to find ``metric_name{label="value", label!="value"}``
# fragments and pull the label/value pairs out. A full parser is
# unnecessary here — Prometheus rules don't nest matchers, and the
# expressions in the bundled set use the standard literal shape. The
# regex below tolerates whitespace between matchers and across
# multi-line ``expr:`` blocks (the ``|`` style used in YAML).
# ---------------------------------------------------------------------------

_FORECAST_NAME = re.compile(r"\b([a-zA-Z_:][a-zA-Z0-9_:]*)\s*\{")
_MATCHER = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(!=|=~|!~|=)\s*"([^"]*)"')

# Forecaster-scrape labels that, when applied as filters to a forecast
# metric, would silently match zero series under the sink-first
# defaults. This list is the canonical "drift detector" — extend it
# only if the snapshot path grows a new scrape-side label.
_SCRAPE_ONLY_VALUES = {
    "instance": re.compile(r"^[^:]*forecaster[^:]*:\d+$|^forecaster:\d+$"),
    "job": re.compile(r"^promforecast$|^.*-promforecast$"),
}


def _iter_rule_exprs() -> list[tuple[str, str, str]]:
    """Yield ``(group_name, alert_name, expr)`` for every alert in the file."""
    doc = yaml.safe_load(RULES_PATH.read_text())
    out: list[tuple[str, str, str]] = []
    for g in doc["spec"]["groups"]:
        for r in g.get("rules", []):
            if "alert" not in r:
                continue
            out.append((g["name"], r["alert"], r["expr"]))
    return out


def _iter_metric_matchers(expr: str) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Pull every ``metric{matchers}`` block from a PromQL expression.

    Returns ``[(metric_name, [(label, op, value), ...])]``. Bare metric
    references (no ``{...}``) are not returned — they are unconstrained
    and trivially satisfiable against any non-empty fixture entry.
    """
    out: list[tuple[str, list[tuple[str, str, str]]]] = []
    # ``re.finditer`` over the ``metric_name{`` boundary; from each
    # match we walk forward to the matching closing brace and slice the
    # selector block. This avoids the regex-engine pathology of trying
    # to match nested braces.
    for m in _FORECAST_NAME.finditer(expr):
        name = m.group(1)
        # Skip language keywords that can immediately precede ``{`` in
        # function-style calls (none of the bundled rules use ``by({...})``
        # but ``without({...})`` is a real construct).
        if name in {"by", "without", "on", "ignoring", "group_left", "group_right"}:
            continue
        start = m.end()
        depth = 1
        end = start
        while end < len(expr) and depth > 0:
            ch = expr[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            end += 1
        if depth != 0:
            continue
        block = expr[start : end - 1]
        matchers = [(lbl, op, val) for lbl, op, val in _MATCHER.findall(block)]
        out.append((name, matchers))
    return out


def _labelset_matches(labels: dict[str, str], matchers: list[tuple[str, str, str]]) -> bool:
    """Return True if ``labels`` would satisfy every matcher in ``matchers``."""
    for label, op, value in matchers:
        present = labels.get(label, "")
        if op == "=":
            if present != value:
                return False
        elif op == "!=":
            if present == value:
                return False
        elif op == "=~":
            if not re.fullmatch(value, present):
                return False
        elif op == "!~" and re.fullmatch(value, present):
            return False
    return True


@pytest.mark.parametrize(("group", "alert", "expr"), _iter_rule_exprs())
def test_rule_metric_names_avoid_reserved_colons(group: str, alert: str, expr: str) -> None:
    """Forecaster output metric names must never use colons.

    The project's naming convention reserves ``:`` for Prometheus
    recording rules — exporters and direct instrumentation never use it.
    ``exporter.py`` rejects colons at runtime, but bundled alerts could
    drift independently (a recording-rule expression copy-pasted here
    by mistake). This test catches that at PR time.
    """
    for metric, _ in _iter_metric_matchers(expr):
        if ":" in metric:
            raise AssertionError(
                f"{group}/{alert}: rule references metric {metric!r} which uses the "
                "reserved ``:`` character. Forecaster output metrics never use colons "
                "(forecaster naming convention). Did a recording-rule expression "
                "slip into the bundled alerts?"
            )


@pytest.mark.parametrize(("group", "alert", "expr"), _iter_rule_exprs())
def test_rule_uses_sink_compatible_label_filters(group: str, alert: str, expr: str) -> None:
    """No example rule should pin a forecast metric to a scrape-only label.

    A pin like ``instance="forecaster:9091"`` on ``*_forecast`` would have
    matched the snapshot scrape but matches zero series under the sink+VM
    path. Operational metrics (``forecast_*``,
    ``*_forecast_time_to_threshold_seconds``) are allowed to filter on the
    scrape labels because they live only on /metrics.
    """
    for metric, matchers in _iter_metric_matchers(expr):
        if not re.fullmatch(r".+_forecast(_lower|_upper)?", metric):
            continue
        for label, op, value in matchers:
            if op != "=":
                continue
            pattern = _SCRAPE_ONLY_VALUES.get(label)
            if pattern and pattern.fullmatch(value):
                raise AssertionError(
                    f"{group}/{alert}: rule pins sink-emitted metric {metric!r} "
                    f'on snapshot-only label {label}="{value}" — under the sink+VM '
                    "defaults this would silently match zero series."
                )


@pytest.mark.parametrize(("group", "alert", "expr"), _iter_rule_exprs())
def test_rule_metrics_resolve_against_sink_fixture(group: str, alert: str, expr: str) -> None:
    """Every leaf metric reference should bind to at least one fixture series.

    **Scope: this is a binding test, not a correctness test.** It asserts
    that each matcher resolves to at least one series in the sink-shaped
    fixture — enough to catch typos (a metric renamed but the alert not
    refreshed) and label-filter drift away from the sink labelset (a
    matcher pinning a label value the sink never projects). It does *not*
    audit alert semantics: a rule like ``count by (group) (X) > 0`` will
    pass this test even if a label dimension (``cold_start``,
    ``model_fallback``, ``level``) is silently aggregated over. Treat the
    test as a guard against label drift, not as a substitute for review.
    """
    skip_alerts = {
        # Capacity templates are deliberately written for operator-tuned
        # ``thresholds:`` configs the dev stack does not enable. The
        # fixture would balloon needlessly trying to model every shape;
        # the rule pinning above already audits the labelset.
    }
    if alert in skip_alerts:
        pytest.skip("environment-specific template")
    for metric, matchers in _iter_metric_matchers(expr):
        series = FIXTURE.get(metric)
        if series is None:
            raise AssertionError(
                f"{group}/{alert}: rule references {metric!r} but the sink-shaped "
                "test fixture has no series for it — extend FIXTURE in this file "
                "when adding a new emission family, or fix the rule if it drifted."
            )
        if not any(_labelset_matches(ls, matchers) for ls in series):
            raise AssertionError(
                f"{group}/{alert}: matcher block on {metric!r} matches zero "
                f"fixture series — expr={expr!r}"
            )


def test_header_declares_sink_assumption() -> None:
    """The file's leading comment must call out the sink+VM assumption.

    Operators who paste a single rule out of this file lose the comment;
    the assumption is still recoverable from the docs link the header
    points at. Drift here means the header was edited; refresh both
    halves so the docs reference stays accurate.
    """
    head = RULES_PATH.read_text(encoding="utf-8").splitlines()[:20]
    joined = "\n".join(head)
    assert "sink+VM" in joined or "sink + VM" in joined, joined
    assert "docs/reference/emission-paths.md" in joined, joined
