"""Tests for the ``incident_pattern`` module and the runner's event-profile path."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from promforecast import incident_pattern
from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter, IncidentRecurrencePoint
from promforecast.source import PromSource, SeriesFrame

# ---- counter_to_event_counts -----------------------------------------------


def _ts(*offsets_minutes: int) -> list[datetime]:
    base = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)  # Monday 00:00 UTC
    return [base + timedelta(minutes=m) for m in offsets_minutes]


def test_counter_to_event_counts_handles_monotonic_counter() -> None:
    timestamps = _ts(0, 5, 10, 15)
    values = [0.0, 1.0, 1.0, 3.0]
    out = incident_pattern.counter_to_event_counts(timestamps=timestamps, values=values)
    # Three deltas: +1 at idx 1, 0 at idx 2, +2 at idx 3.
    # Zero deltas are excluded.
    assert [(t, c) for t, c in out] == [
        (timestamps[1], 1),
        (timestamps[3], 2),
    ]


def test_counter_to_event_counts_drops_negative_deltas() -> None:
    """Counter resets must be ignored, not emitted as negative events."""
    timestamps = _ts(0, 5, 10)
    values = [5.0, 0.0, 1.0]  # reset between idx 0 and idx 1
    out = incident_pattern.counter_to_event_counts(timestamps=timestamps, values=values)
    # The negative delta is dropped; the +1 delta is preserved.
    assert [(t, c) for t, c in out] == [(timestamps[2], 1)]


def test_counter_to_event_counts_handles_nan_samples() -> None:
    timestamps = _ts(0, 5, 10)
    values = [1.0, math.nan, 2.0]
    out = incident_pattern.counter_to_event_counts(timestamps=timestamps, values=values)
    # Both deltas involving NaN are dropped — leaves the empty list.
    assert out == []


def test_counter_to_event_counts_short_input() -> None:
    assert incident_pattern.counter_to_event_counts(timestamps=_ts(0), values=[1.0]) == []


def test_counter_to_event_counts_rounds_fractional_deltas() -> None:
    """Counters that tick fractionally (rate-derived input) round to int."""
    timestamps = _ts(0, 5)
    values = [0.0, 1.6]
    out = incident_pattern.counter_to_event_counts(timestamps=timestamps, values=values)
    assert out == [(timestamps[1], 2)]


# ---- bucket_events_by_week_hour --------------------------------------------


def test_bucket_returns_full_7x24_grid_for_empty_input() -> None:
    grid = incident_pattern.bucket_events_by_week_hour(events=[])
    assert len(grid) == 7 * 24
    assert all(b.count == 0 and b.score == 0.0 for b in grid)


def test_bucket_normalises_against_non_empty_mean() -> None:
    """Bucket score = count / mean(populated buckets)."""
    # Two events at Monday 0h, four events at Monday 1h.
    # populated buckets = 2; mean = (2+4)/2 = 3.
    # Monday 0h scores 2/3, Monday 1h scores 4/3.
    monday_0h = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    monday_1h = datetime(2026, 5, 18, 1, 0, tzinfo=UTC)
    events = [(monday_0h, 2), (monday_1h, 4)]
    grid = incident_pattern.bucket_events_by_week_hour(events=events)
    by_key = {(b.day_of_week, b.hour_of_day): b for b in grid}
    assert by_key[(0, 0)].score == pytest.approx(2 / 3)
    assert by_key[(0, 1)].score == pytest.approx(4 / 3)
    # Other buckets stay at zero.
    assert by_key[(0, 2)].score == 0.0
    assert by_key[(3, 14)].score == 0.0


def test_bucket_single_event_scores_one() -> None:
    """A single observed event is the mean: score = 1.0."""
    events = [(datetime(2026, 5, 18, 9, 0, tzinfo=UTC), 1)]
    grid = incident_pattern.bucket_events_by_week_hour(events=events)
    by_key = {(b.day_of_week, b.hour_of_day): b for b in grid}
    assert by_key[(0, 9)].score == 1.0
    assert by_key[(0, 9)].count == 1


def test_bucket_uses_iso_weekday_monday_zero() -> None:
    """Monday = 0, Sunday = 6 (ISO convention)."""
    sun = datetime(2026, 5, 17, 14, 0, tzinfo=UTC)  # Sunday
    grid = incident_pattern.bucket_events_by_week_hour(events=[(sun, 1)])
    by_key = {(b.day_of_week, b.hour_of_day): b for b in grid}
    assert by_key[(6, 14)].score == 1.0


def test_compute_recurrence_end_to_end() -> None:
    """Wrapper: counter → buckets in one call."""
    # Counter ticks once at Tuesday 03:00, twice at Tuesday 04:00.
    base = datetime(2026, 5, 19, 3, 0, tzinfo=UTC)  # Tuesday
    timestamps = [base, base + timedelta(hours=1), base + timedelta(hours=2)]
    values = [0.0, 1.0, 3.0]
    grid = incident_pattern.compute_recurrence(timestamps=timestamps, values=values)
    by_key = {(b.day_of_week, b.hour_of_day): b for b in grid}
    # Tuesday = 1; events at 04:00 (+1) and 05:00 (+2). Two populated
    # buckets, mean = 1.5. 04:00 scores 1/1.5; 05:00 scores 2/1.5.
    assert by_key[(1, 4)].score == pytest.approx(1 / 1.5)
    assert by_key[(1, 5)].score == pytest.approx(2 / 1.5)


# ---- Exporter integration --------------------------------------------------


def test_exporter_renders_incident_recurrence() -> None:
    exporter = Exporter()
    exporter.replace_incident_recurrence_for_query(
        group="g",
        query="incidents",
        points=[
            IncidentRecurrencePoint(
                labels={"group": "g", "id": "incidents", "day_of_week": "0", "hour_of_day": "9"},
                score=2.5,
            )
        ],
    )
    body = exporter.render().decode()
    assert "forecast_incident_recurrence_score" in body
    assert 'day_of_week="0"' in body
    assert 'hour_of_day="9"' in body


def test_exporter_replaces_per_query_state() -> None:
    """Replacing one query preserves other queries' rows."""
    exporter = Exporter()
    exporter.replace_incident_recurrence_for_query(
        group="g",
        query="a",
        points=[
            IncidentRecurrencePoint(
                labels={"group": "g", "id": "a", "day_of_week": "0", "hour_of_day": "0"},
                score=1.0,
            )
        ],
    )
    exporter.replace_incident_recurrence_for_query(
        group="g",
        query="b",
        points=[
            IncidentRecurrencePoint(
                labels={"group": "g", "id": "b", "day_of_week": "0", "hour_of_day": "0"},
                score=2.0,
            )
        ],
    )
    # Both queries' rows present.
    body = exporter.render().decode()
    assert 'id="a"' in body
    assert 'id="b"' in body
    # Replacing query 'a' clears its rows but keeps 'b'.
    exporter.replace_incident_recurrence_for_query(group="g", query="a", points=[])
    body = exporter.render().decode()
    assert 'id="a"' not in body
    assert 'id="b"' in body


def test_retain_incident_recurrence_groups_drops_removed_groups() -> None:
    exporter = Exporter()
    exporter.replace_incident_recurrence_for_query(
        group="dropped",
        query="q",
        points=[
            IncidentRecurrencePoint(
                labels={"group": "dropped", "id": "q", "day_of_week": "0", "hour_of_day": "0"},
                score=1.0,
            )
        ],
    )
    exporter.replace_incident_recurrence_for_query(
        group="kept",
        query="q",
        points=[
            IncidentRecurrencePoint(
                labels={"group": "kept", "id": "q", "day_of_week": "0", "hour_of_day": "0"},
                score=1.0,
            )
        ],
    )
    exporter.retain_incident_recurrence_groups({"kept"})
    body = exporter.render().decode()
    assert 'group="dropped"' not in body
    assert 'group="kept"' in body


# ---- Runner wiring ---------------------------------------------------------


class _StubSource(PromSource):
    """Returns a synthetic counter series for the event-profile test."""

    def __init__(self, frames: list[SeriesFrame] | None = None) -> None:
        self._frames = frames

    async def query_range(
        self, promql: str, start: datetime, end: datetime, step_seconds: int
    ) -> list[SeriesFrame]:
        if self._frames is None:
            # Build a synthetic Monday-08:00 single-event counter.
            base = datetime(2026, 5, 18, 8, 0, tzinfo=UTC)
            timestamps = [base + timedelta(minutes=5 * i) for i in range(3)]
            values = [0.0, 0.0, 1.0]
            return [
                SeriesFrame(
                    labels={"severity": "critical"},
                    timestamps=timestamps,
                    values=values,
                )
            ]
        return list(self._frames)

    async def aclose(self) -> None:
        return None


def _events_config() -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            lookback=timedelta(hours=1),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[
            GroupConfig(
                name="incidents",
                queries=[
                    QueryConfig(
                        id="pagerduty_incidents_total",
                        promql="pagerduty_incidents_total",
                        data_profile="events",
                    )
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_runner_event_profile_emits_recurrence_grid() -> None:
    from promforecast.runner import Runner  # noqa: PLC0415

    cfg = _events_config()
    exporter = Exporter()
    runner = Runner(config=cfg, source=_StubSource(), exporter=exporter)
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert "forecast_incident_recurrence_score" in body
    # 7 * 24 = 168 buckets emitted for the single configured query.
    assert body.count("forecast_incident_recurrence_score{") == 7 * 24


@pytest.mark.asyncio
async def test_runner_event_profile_handles_empty_source() -> None:
    """An empty source response clears the heatmap rather than crashing."""
    from promforecast.runner import Runner  # noqa: PLC0415

    cfg = _events_config()
    exporter = Exporter()
    runner = Runner(config=cfg, source=_StubSource(frames=[]), exporter=exporter)
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert "forecast_incident_recurrence_score" not in body


@pytest.mark.asyncio
async def test_runner_event_profile_handles_source_failure() -> None:
    """Source failure increments forecast_failures_total without crashing."""
    from promforecast.runner import Runner  # noqa: PLC0415

    class _RaisingSource(PromSource):
        def __init__(self) -> None:
            pass

        async def query_range(self, *a: object, **k: object) -> list[SeriesFrame]:
            raise RuntimeError("source down")

        async def aclose(self) -> None:
            return None

    cfg = _events_config()
    exporter = Exporter()
    runner = Runner(config=cfg, source=_RaisingSource(), exporter=exporter)
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    # No recurrence rows emitted on failure.
    assert "forecast_incident_recurrence_score" not in body
    # query_error counter bumped.
    assert 'reason="query_error"' in body


# ---- module surface --------------------------------------------------------


def test_data_profile_events_rejects_thresholds(tmp_path: object) -> None:
    """``data_profile: events`` with thresholds fails fast at load time
    rather than silently dropping the threshold the operator
    configured (it would never fire because the event path emits no
    forecast curve)."""
    from promforecast.config import (  # noqa: PLC0415
        QueryConfig as _QueryConfig,
    )
    from promforecast.config import (  # noqa: PLC0415
        ThresholdConfig as _ThresholdConfig,
    )
    from promforecast.config import (  # noqa: PLC0415
        _validate_query_overrides,
    )

    query = _QueryConfig(
        id="incidents",
        promql="pagerduty_incidents_total",
        data_profile="events",
        thresholds=[_ThresholdConfig(name="full", value=10.0, comparator="gt")],
    )
    with pytest.raises(ValueError, match="thresholds"):
        _validate_query_overrides(group_name="g", query=query)


def test_data_profile_events_rejects_regressors() -> None:
    from promforecast.config import (  # noqa: PLC0415
        QueryConfig as _QueryConfig,
    )
    from promforecast.config import (  # noqa: PLC0415
        RegressorConfig as _RegressorConfig,
    )
    from promforecast.config import (  # noqa: PLC0415
        _validate_query_overrides,
    )

    query = _QueryConfig(
        id="incidents",
        promql="pagerduty_incidents_total",
        data_profile="events",
        regressors=[_RegressorConfig(id="deploys", promql="up")],
    )
    with pytest.raises(ValueError, match="regressors"):
        _validate_query_overrides(group_name="g", query=query)


def test_data_profile_events_alone_is_accepted() -> None:
    """The events profile on its own (no thresholds / regressors /
    SLO / etc.) must validate cleanly."""
    from promforecast.config import (  # noqa: PLC0415
        QueryConfig as _QueryConfig,
    )
    from promforecast.config import (  # noqa: PLC0415
        _validate_query_overrides,
    )

    query = _QueryConfig(
        id="incidents",
        promql="pagerduty_incidents_total",
        data_profile="events",
    )
    _validate_query_overrides(group_name="g", query=query)


@pytest.mark.asyncio
async def test_reload_drops_heatmap_for_removed_event_query() -> None:
    """A reload that removes the events query from a still-present group
    must drop the heatmap rows; otherwise the 168 stale rows linger on
    /metrics until process restart."""
    from promforecast.runner import Runner  # noqa: PLC0415

    cfg = _events_config()
    exporter = Exporter()
    runner = Runner(config=cfg, source=_StubSource(), exporter=exporter)
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert "forecast_incident_recurrence_score" in body
    # Reload with the events query removed but the group retained.
    new_cfg = Config(
        datasource=cfg.datasource,
        server=cfg.server,
        safety=cfg.safety,
        defaults=cfg.defaults,
        groups=[
            GroupConfig(
                name="incidents",
                queries=[
                    QueryConfig(
                        id="other_metric",
                        promql="up",
                    )
                ],
            )
        ],
    )
    await runner.apply_config(new_cfg)
    body = exporter.render().decode()
    assert "forecast_incident_recurrence_score" not in body
    await runner.stop()


def test_exporter_retain_incident_recurrence_queries_drops_removed_query() -> None:
    """Per-(group, query) retain handles a single-query removal."""
    exporter = Exporter()
    for query_id in ("kept_query", "removed_query"):
        exporter.replace_incident_recurrence_for_query(
            group="g",
            query=query_id,
            points=[
                IncidentRecurrencePoint(
                    labels={
                        "group": "g",
                        "id": query_id,
                        "day_of_week": "0",
                        "hour_of_day": "0",
                    },
                    score=1.0,
                )
            ],
        )
    exporter.retain_incident_recurrence_queries({("g", "kept_query")})
    body = exporter.render().decode()
    assert 'id="kept_query"' in body
    assert 'id="removed_query"' not in body


def test_module_exports_documented_names() -> None:
    assert incident_pattern.compute_recurrence is not None
    assert incident_pattern.counter_to_event_counts is not None
    assert incident_pattern.bucket_events_by_week_hour is not None
    assert incident_pattern.RecurrenceBucket is not None
