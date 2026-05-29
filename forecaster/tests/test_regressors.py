"""Tests for the two-phase regressor resolver.

* :func:`prefetch_promql` runs once per primary query — covered here for
  graceful degradation, the required-regressor short-circuit, and the
  no-source path.
* :func:`align_for_series` runs per series — covered here for calendar
  feature derivation, nearest-asof alignment, and the missing-prefetch
  optional path.

The end-to-end runner integration (per-query call counts, counter
semantics) is exercised in ``test_runner_v03.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from promforecast.config import RegressorConfig
from promforecast.regressors import (
    PromQLPrefetch,
    RegressorError,
    align_for_series,
    prefetch_promql,
)
from promforecast.source import PromSource, SeriesFrame
from promforecast.webhook import WebhookError


class _FakeSource(PromSource):
    """Records call counts so we can assert PromQL fan-out economy."""

    def __init__(
        self,
        responses: dict[str, list[SeriesFrame] | Exception] | None = None,
    ) -> None:
        self._responses: dict[str, list[SeriesFrame] | Exception] = responses or {}
        self.call_count: dict[str, int] = {}

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.call_count[promql] = self.call_count.get(promql, 0) + 1
        result = self._responses.get(promql, [])
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self) -> None:
        return None


def _hourly_timestamps(n: int, *, start: datetime | None = None) -> list[datetime]:
    base = start or datetime(2024, 6, 3, 12, 0, tzinfo=UTC)  # Monday at noon
    return [base + timedelta(hours=i) for i in range(n)]


# -- prefetch_promql --------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_returns_empty_when_no_promql_regressors() -> None:
    pf = await prefetch_promql(
        regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
        source=None,
        query_start=datetime.now(tz=UTC),
        query_end=datetime.now(tz=UTC),
        step_seconds=60,
        query_timeout_seconds=5.0,
    )
    assert pf.fetched == {}
    assert pf.failures == []


@pytest.mark.asyncio
async def test_prefetch_optional_failure_recorded_once() -> None:
    source = _FakeSource({"deploys": RuntimeError("boom")})
    pf = await prefetch_promql(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
        source=source,
        query_start=datetime.now(tz=UTC),
        query_end=datetime.now(tz=UTC),
        step_seconds=60,
        query_timeout_seconds=5.0,
    )
    assert pf.fetched == {}
    assert [f.reason for f in pf.failures] == ["query_error"]
    assert source.call_count == {"deploys": 1}


@pytest.mark.asyncio
async def test_prefetch_required_failure_raises() -> None:
    source = _FakeSource({"deploys": RuntimeError("boom")})
    with pytest.raises(RegressorError) as excinfo:
        await prefetch_promql(
            regressors=[
                RegressorConfig(id="deploys", type="custom", promql="deploys", required=True)
            ],
            source=source,
            query_start=datetime.now(tz=UTC),
            query_end=datetime.now(tz=UTC),
            step_seconds=60,
            query_timeout_seconds=5.0,
        )
    assert excinfo.value.reason == "query_error"


@pytest.mark.asyncio
async def test_prefetch_no_source_treated_as_failure_per_regressor() -> None:
    pf = await prefetch_promql(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
        source=None,
        query_start=datetime.now(tz=UTC),
        query_end=datetime.now(tz=UTC),
        step_seconds=60,
        query_timeout_seconds=5.0,
    )
    assert pf.fetched == {}
    assert [f.reason for f in pf.failures] == ["no_source"]


@pytest.mark.asyncio
async def test_prefetch_empty_result_is_a_failure() -> None:
    source = _FakeSource({"deploys": []})
    pf = await prefetch_promql(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
        source=source,
        query_start=datetime.now(tz=UTC),
        query_end=datetime.now(tz=UTC),
        step_seconds=60,
        query_timeout_seconds=5.0,
    )
    assert pf.fetched == {}
    assert [f.reason for f in pf.failures] == ["empty_result"]


# -- align_for_series ------------------------------------------------------


def test_align_returns_none_when_no_regressors() -> None:
    out = align_for_series(
        regressors=[],
        prefetch=PromQLPrefetch(fetched={}, failures=[]),
        timestamps=_hourly_timestamps(3),
    )
    assert out.frame is None
    assert out.failures == []


def test_align_calendar_features_are_deterministic() -> None:
    out = align_for_series(
        regressors=[
            RegressorConfig(id="hour_of_day", type="calendar"),
            RegressorConfig(id="day_of_week", type="calendar"),
            RegressorConfig(id="is_weekend", type="calendar"),
        ],
        prefetch=PromQLPrefetch(fetched={}, failures=[]),
        timestamps=_hourly_timestamps(3),
    )
    assert out.frame is not None
    assert list(out.frame["hour_of_day"]) == [12.0, 13.0, 14.0]
    assert list(out.frame["day_of_week"]) == [0.0, 0.0, 0.0]  # Monday
    assert list(out.frame["is_weekend"]) == [0.0, 0.0, 0.0]


def test_align_uses_prefetched_promql_with_nearest_asof() -> None:
    timestamps = _hourly_timestamps(3)
    # Source data is shifted by 10 minutes; nearest-asof should still
    # produce one row per query timestamp.
    frame = SeriesFrame(
        labels={"job": "deployer"},
        timestamps=[t + timedelta(minutes=10) for t in timestamps],
        values=[1.0, 2.0, 3.0],
    )
    pf = PromQLPrefetch(fetched={"deploys": [frame]}, failures=[])
    out = align_for_series(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
        prefetch=pf,
        timestamps=timestamps,
    )
    assert out.frame is not None
    assert list(out.frame["deploys"]) == [1.0, 2.0, 3.0]


def test_align_drops_optional_promql_missing_from_prefetch() -> None:
    """An optional regressor whose pre-fetch failed isn't in
    ``prefetch.fetched``; its column is silently omitted (the failure has
    already been counted at prefetch time)."""
    out = align_for_series(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
        prefetch=PromQLPrefetch(fetched={}, failures=[]),
        timestamps=_hourly_timestamps(3),
    )
    assert out.frame is None  # only ``ds`` column → dropped
    assert out.failures == []  # PromQL failure was counted at prefetch


def test_align_mixes_calendar_and_promql() -> None:
    timestamps = _hourly_timestamps(2)
    frame = SeriesFrame(labels={}, timestamps=timestamps, values=[10.0, 20.0])
    pf = PromQLPrefetch(fetched={"deploys": [frame]}, failures=[])
    out = align_for_series(
        regressors=[
            RegressorConfig(id="hour_of_day", type="calendar"),
            RegressorConfig(id="deploys", type="custom", promql="deploys"),
        ],
        prefetch=pf,
        timestamps=timestamps,
    )
    assert out.frame is not None
    assert list(out.frame["hour_of_day"]) == [12.0, 13.0]
    assert list(out.frame["deploys"]) == [10.0, 20.0]


# -- webhook regressors (prefetch wiring) ----------------------------------


class _FakeWebhookClient:
    """Returns canned frames (or raises) without touching the network."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.calls = 0

    async def fetch(
        self,
        *,
        reg: RegressorConfig,
        query_id: str,
        start: datetime,
        end: datetime,
    ) -> list[SeriesFrame]:
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result  # type: ignore[return-value]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_prefetch_webhook_success_is_aligned() -> None:
    timestamps = _hourly_timestamps(2)
    frame = SeriesFrame(
        labels={"regressor_id": "incidents"}, timestamps=timestamps, values=[1.0, 2.0]
    )
    client = _FakeWebhookClient([frame])
    reg = RegressorConfig(id="incidents", type="webhook", url="https://hook/x")
    pf = await prefetch_promql(
        regressors=[reg],
        source=None,
        query_start=timestamps[0],
        query_end=timestamps[-1],
        step_seconds=3600,
        query_timeout_seconds=5.0,
        query_id="cpu",
        webhook_client=client,  # type: ignore[arg-type]
    )
    assert client.calls == 1
    out = align_for_series(regressors=[reg], prefetch=pf, timestamps=timestamps)
    assert out.frame is not None
    assert list(out.frame["incidents"]) == [1.0, 2.0]


@pytest.mark.asyncio
async def test_prefetch_webhook_optional_failure_degrades() -> None:
    client = _FakeWebhookClient(WebhookError("webhook_failure", "boom"))
    reg = RegressorConfig(id="incidents", type="webhook", url="https://hook/x")
    pf = await prefetch_promql(
        regressors=[reg],
        source=None,
        query_start=datetime.now(tz=UTC),
        query_end=datetime.now(tz=UTC),
        step_seconds=3600,
        query_timeout_seconds=5.0,
        query_id="cpu",
        webhook_client=client,  # type: ignore[arg-type]
    )
    assert pf.fetched == {}
    assert [f.reason for f in pf.failures] == ["webhook_failure"]


@pytest.mark.asyncio
async def test_prefetch_webhook_required_failure_raises() -> None:
    client = _FakeWebhookClient(WebhookError("signature_invalid", "bad sig"))
    reg = RegressorConfig(id="incidents", type="webhook", url="https://hook/x", required=True)
    with pytest.raises(RegressorError) as excinfo:
        await prefetch_promql(
            regressors=[reg],
            source=None,
            query_start=datetime.now(tz=UTC),
            query_end=datetime.now(tz=UTC),
            step_seconds=3600,
            query_timeout_seconds=5.0,
            query_id="cpu",
            webhook_client=client,  # type: ignore[arg-type]
        )
    assert excinfo.value.reason == "signature_invalid"


@pytest.mark.asyncio
async def test_prefetch_webhook_no_client_degrades() -> None:
    reg = RegressorConfig(id="incidents", type="webhook", url="https://hook/x")
    pf = await prefetch_promql(
        regressors=[reg],
        source=None,
        query_start=datetime.now(tz=UTC),
        query_end=datetime.now(tz=UTC),
        step_seconds=3600,
        query_timeout_seconds=5.0,
        query_id="cpu",
        webhook_client=None,
    )
    assert [f.reason for f in pf.failures] == ["no_source"]


# -- calendar_file regressors (per-series alignment) -----------------------


def _write_calendar(tmp_path: object, body: str) -> str:
    path = Path(str(tmp_path)) / "freezes.yaml"
    path.write_text(body)
    return str(path)


def test_align_calendar_file_binary(tmp_path: object) -> None:
    path = _write_calendar(
        tmp_path,
        "entries:\n  - start: 2024-06-03\n    end: 2024-06-03\n    name: freeze\n",
    )
    # _hourly_timestamps starts 2024-06-03 12:00 UTC; 2 points same day.
    timestamps = _hourly_timestamps(2)
    reg = RegressorConfig(id="release_freeze", type="calendar_file", path=path)
    out = align_for_series(
        regressors=[reg],
        prefetch=PromQLPrefetch(fetched={}, failures=[]),
        timestamps=timestamps,
    )
    assert out.frame is not None
    assert list(out.frame["release_freeze"]) == [1.0, 1.0]


def test_align_calendar_file_expand_one_hot(tmp_path: object) -> None:
    path = _write_calendar(
        tmp_path,
        "entries:\n"
        "  - start: 2024-06-03\n    end: 2024-06-03\n    name: freeze-a\n"
        "  - start: 2030-01-01\n    end: 2030-01-01\n    name: freeze-b\n",
    )
    timestamps = _hourly_timestamps(1)
    reg = RegressorConfig(id="cal", type="calendar_file", path=path, expand=True)
    out = align_for_series(
        regressors=[reg],
        prefetch=PromQLPrefetch(fetched={}, failures=[]),
        timestamps=timestamps,
    )
    assert out.frame is not None
    assert list(out.frame["cal__freeze_a"]) == [1.0]
    assert list(out.frame["cal__freeze_b"]) == [0.0]


def test_align_calendar_file_missing_file_degrades() -> None:
    reg = RegressorConfig(id="cal", type="calendar_file", path="/nope/missing.yaml")
    out = align_for_series(
        regressors=[reg],
        prefetch=PromQLPrefetch(fetched={}, failures=[]),
        timestamps=_hourly_timestamps(2),
    )
    assert out.frame is None
    assert [f.reason for f in out.failures] == ["calendar_file_error"]
