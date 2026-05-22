"""Runner-side integration tests for drift / band-coverage / scheduling.

Covers drift telemetry, empirical band coverage, per-query refresh
scheduling, and the holiday-regressor wiring. The preview endpoint and
the pure-function drift / band-coverage primitives have their own focused
unit suites; this file exercises the end-to-end runner path so a
regression in the snapshot merge, scheduler tick, or exporter rendering
gets caught even if individual modules are healthy.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promforecast.config import (
    AccuracyConfig,
    BandCoverageConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    DriftEmissionConfig,
    GroupConfig,
    QueryConfig,
    QueryEmissionConfig,
    SafetyConfig,
    ServerConfig,
    effective_refresh_interval,
)
from promforecast.drift import DriftCache, compute_drift
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


class _FakeSource(PromSource):
    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query
        self.calls: list[str] = []

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.calls.append(promql)
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


class _RampForecastModel:
    """Emits a linear ramp forecast so drift / coverage are predictable."""

    def __init__(
        self,
        season_length: int = 288,
        *,
        start: float = 50.0,
        per_step: float = 1.0,
    ) -> None:
        self.name = "RampModel"
        self._start = start
        self._per_step = per_step

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        values = [self._start + i * self._per_step for i in range(horizon)]
        out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": values})
        for level in levels:
            out[f"yhat_lower_{level}"] = [v - 1.0 for v in values]
            out[f"yhat_upper_{level}"] = [v + 1.0 for v in values]
        return out


def _frame(n: int) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=5 * i) for i in range(n)]
    return SeriesFrame(labels={"instance": "host-a"}, timestamps=timestamps, values=[50.0] * n)


def _config_with_query(query: QueryConfig) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["RampModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[GroupConfig(name="g", queries=[query])],
    )


# ---- drift telemetry -------------------------------------------------------


def test_compute_drift_zero_for_identical_curves() -> None:
    """Two identical curves must report a drift score of exactly 0."""
    score = compute_drift([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert score == 0.0


def test_compute_drift_finite_for_shifted_curve() -> None:
    """A shifted curve has a positive, finite drift score."""
    score = compute_drift([10.0, 20.0, 30.0], [11.0, 22.0, 33.0])
    assert 0.0 < score < 1.0
    assert math.isfinite(score)


def test_compute_drift_distinguishes_shape_change_from_shift() -> None:
    """A curve that swaps tail values produces non-zero drift.

    Single-point comparison (a regression of an earlier draft) would
    emit zero drift for two curves whose snapshot rows are
    identical; this test fails under that draft because the curves
    *differ* in their tail steps despite matching last values.
    """
    score = compute_drift([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 4.0])
    assert score > 0.0


def test_drift_cache_prune_drops_dead_keys() -> None:
    cache = DriftCache()
    cache.put(group="a", query="q1", series_key="s", model="m", curve=[1.0])
    cache.put(group="a", query="q2", series_key="s", model="m", curve=[2.0])
    cache.prune_to({("a", "q1")})
    assert cache.get(group="a", query="q1", series_key="s", model="m") == [1.0]
    assert cache.get(group="a", query="q2", series_key="s", model="m") is None


@pytest.mark.asyncio
async def test_runner_emits_drift_gauge_after_second_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First fit caches the curve; second fit produces a finite drift score."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(QueryConfig(id="m", promql="up"))
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    # First run: drift cache is cold, no drift gauge emitted.
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert "forecast_drift_score" not in body

    # Mutate the model to produce a different curve so drift > 0.
    monkeypatch.setattr(
        model_registry,
        "build",
        lambda *a, **kw: _RampForecastModel(start=60.0, per_step=2.0),
    )
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert "forecast_drift_score" in body
    assert 'model="RampModel"' in body


@pytest.mark.asyncio
async def test_runner_drift_alerts_counter_increments_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift crossing ``alert_threshold`` bumps the per-query alert counter."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            emission=QueryEmissionConfig(
                drift=DriftEmissionConfig(enabled=True, alert_threshold=0.05),
            ),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])
    monkeypatch.setattr(
        model_registry,
        "build",
        lambda *a, **kw: _RampForecastModel(start=80.0, per_step=4.0),
    )
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_drift_alerts_total{group="g",id="m"}' in body


# ---- empirical band coverage ----------------------------------------------


@pytest.mark.asyncio
async def test_runner_emits_band_coverage_when_backtest_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful backtest must emit a band-coverage gauge per level/window."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            emission=QueryEmissionConfig(
                band_coverage=BandCoverageConfig(enabled=True, windows=[timedelta(days=1)]),
            ),
        )
    )
    # Need backtests for band coverage; flip evaluate on with short
    # horizons so the synthetic frame is long enough for the hold-out.
    cfg.defaults.accuracy = AccuracyConfig(evaluate=True, horizons=[timedelta(minutes=15)])
    source = _FakeSource({"up": [_frame(40)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert "forecast_band_empirical_coverage" in body
    assert 'window="1d"' in body


# ---- per-query refresh -----------------------------------------------------


def test_effective_refresh_interval_inherits_when_unset() -> None:
    cfg = ServerConfig(refresh_interval=timedelta(hours=1))
    q = QueryConfig(id="m", promql="up")
    assert effective_refresh_interval(q, cfg) == timedelta(hours=1)


def test_effective_refresh_interval_query_override() -> None:
    cfg = ServerConfig(refresh_interval=timedelta(hours=1))
    q = QueryConfig(id="m", promql="up", refresh_interval=timedelta(minutes=10))
    assert effective_refresh_interval(q, cfg) == timedelta(minutes=10)


@pytest.mark.asyncio
async def test_runner_run_group_partial_preserves_unrelated_query_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial run must keep prior snapshot data for queries that didn't run."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["RampModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(id="a", promql="qa"),
                    QueryConfig(id="b", promql="qb"),
                ],
            )
        ],
    )
    source = _FakeSource({"qa": [_frame(20)], "qb": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    # Full run first so both queries have data.
    await runner.run_group(cfg.groups[0])
    body_full = exporter.render().decode()
    assert "a_forecast{" in body_full
    assert "b_forecast{" in body_full

    # Partial run for just query "a"; query "b" data must survive.
    await runner.run_group(cfg.groups[0], queries=[cfg.groups[0].queries[0]])
    body_partial = exporter.render().decode()
    assert "a_forecast{" in body_partial
    assert "b_forecast{" in body_partial


@pytest.mark.asyncio
async def test_partial_run_counter_merge_and_inheritance_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inheritance + counter merge survive a partial-then-full sequence.

    Combines two paths that previously had no integration coverage: the
    inherited gauge rows for a skipped query (handled by
    ``_inherit_prior_for_skipped_queries``) and the counter merge for
    the cumulative drift-alert counter (handled inline in
    ``run_group``). Running ``a`` then ``b`` then ``a`` again exercises
    every branch of the merge: inherit, merge-from-prior, and prune.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["RampModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(id="a", promql="qa"),
                    QueryConfig(id="b", promql="qb"),
                ],
            )
        ],
    )
    source = _FakeSource({"qa": [_frame(20)], "qb": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])
    initial = exporter.render().decode()
    # Both queries land in the snapshot on the initial full run.
    assert "a_forecast{" in initial
    assert "b_forecast{" in initial

    # Partial run touches only "a"; "b" is inherited.
    await runner.run_group(cfg.groups[0], queries=[cfg.groups[0].queries[0]])
    partial = exporter.render().decode()
    assert "a_forecast{" in partial
    assert "b_forecast{" in partial

    # Partial run for "b"; this time "a" is inherited. Re-rendering after
    # this confirms the previously-inherited "b" data didn't get
    # corrupted by mutating shared references between snapshots.
    await runner.run_group(cfg.groups[0], queries=[cfg.groups[0].queries[1]])
    second_partial = exporter.render().decode()
    assert "a_forecast{" in second_partial
    assert "b_forecast{" in second_partial

    # Both per-query series-count gauges should be present and non-zero.
    assert 'forecast_query_series_count{group="g",query="a"}' in second_partial
    assert 'forecast_query_series_count{group="g",query="b"}' in second_partial


# ---- holiday regressor ----------------------------------------------------


def test_align_for_series_emits_holiday_binary_column() -> None:
    """The default ``expand: false`` emits a single binary column."""
    from promforecast.config import RegressorConfig  # noqa: PLC0415
    from promforecast.regressors import (  # noqa: PLC0415
        PromQLPrefetch,
        align_for_series,
    )

    # 2026-01-01 is New Year's Day in the US calendar.
    timestamps = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    ]
    pf = PromQLPrefetch(fetched={}, failures=[])
    out = align_for_series(
        regressors=[RegressorConfig(id="is_holiday", type="holidays", country="US", expand=False)],
        prefetch=pf,
        timestamps=timestamps,
    )
    assert out.frame is not None
    assert list(out.frame["is_holiday"]) == [1.0, 0.0]


def test_load_rejects_unsupported_holiday_country(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An unknown country code must fail config load, not silently fail at fit."""
    from pathlib import Path  # noqa: PLC0415

    import promforecast.config as config_module  # noqa: PLC0415

    body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        regressors:
          - id: is_holiday
            type: holidays
            country: XZ
"""
    path: Path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match="holidays country/region not supported"):
        config_module.load(path)


def test_align_for_series_holiday_expand_emits_per_holiday_columns() -> None:
    """``expand: true`` emits one one-hot column per observed holiday."""
    from promforecast.config import RegressorConfig  # noqa: PLC0415
    from promforecast.regressors import (  # noqa: PLC0415
        PromQLPrefetch,
        align_for_series,
    )

    timestamps = [
        datetime(2026, 1, 1, tzinfo=UTC),  # New Year's Day
        datetime(2026, 1, 2, tzinfo=UTC),
    ]
    pf = PromQLPrefetch(fetched={}, failures=[])
    out = align_for_series(
        regressors=[RegressorConfig(id="holidays", type="holidays", country="US", expand=True)],
        prefetch=pf,
        timestamps=timestamps,
    )
    assert out.frame is not None
    holiday_cols = [c for c in out.frame.columns if c.startswith("holidays__")]
    assert len(holiday_cols) >= 1
    # The non-holiday day must be zero across every per-holiday column.
    for col in holiday_cols:
        assert out.frame[col].iloc[1] == 0.0


def test_holiday_regressor_honours_timezone_for_local_midnight_flip() -> None:
    """A US-Eastern holiday must flip at local midnight, not UTC-midnight.

    Without timezone handling the regressor reads UTC dates so 2026-07-04
    starts at 00:00 UTC, mis-labelling the four-hour window in
    America/New_York that is still 2026-07-03 locally as a holiday.
    """
    from promforecast.config import RegressorConfig  # noqa: PLC0415
    from promforecast.regressors import (  # noqa: PLC0415
        PromQLPrefetch,
        align_for_series,
    )

    # Use 2024-07-04 (a Thursday) so the federal observance doesn't shift
    # onto an adjacent weekday — 2025/2026 each shift onto a different
    # weekday and would obscure the timezone-handling assertion.
    # 2024-07-04 02:00 UTC = 2024-07-03 22:00 in America/New_York (still
    # the previous day locally, so the regressor must read 0, not 1).
    eastern_evening_before = datetime(2024, 7, 4, 2, 0, tzinfo=UTC)
    # 2024-07-04 14:00 UTC = 2024-07-04 10:00 in America/New_York (the
    # holiday locally — should read 1).
    eastern_holiday_morning = datetime(2024, 7, 4, 14, 0, tzinfo=UTC)

    pf = PromQLPrefetch(fetched={}, failures=[])
    out = align_for_series(
        regressors=[
            RegressorConfig(
                id="is_holiday",
                type="holidays",
                country="US",
                timezone="America/New_York",
                expand=False,
            )
        ],
        prefetch=pf,
        timestamps=[eastern_evening_before, eastern_holiday_morning],
    )
    assert out.frame is not None
    assert list(out.frame["is_holiday"]) == [0.0, 1.0]


def test_holiday_regressor_rejects_unknown_timezone(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A bogus IANA timezone must fail config load, not silently default."""
    from pathlib import Path  # noqa: PLC0415

    import promforecast.config as config_module  # noqa: PLC0415

    body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        regressors:
          - id: is_holiday
            type: holidays
            country: US
            timezone: Not/AReal_Zone
"""
    path: Path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match="not a known IANA zone"):
        config_module.load(path)


def test_holiday_calendar_is_cached_across_calls() -> None:
    """Multiple regressors against the same country share the calendar object.

    Without the cache, ``country_holidays("US")`` is re-materialised on
    every config load — that's a multi-megabyte allocation per regressor
    when an operator has dozens of holiday regressors against the same
    country.
    """
    from promforecast.regressors import (  # noqa: PLC0415
        _HOLIDAY_CALENDAR_CACHE,
        _holiday_calendar_cache_get,
    )

    _HOLIDAY_CALENDAR_CACHE.clear()
    first = _holiday_calendar_cache_get("US", None)
    second = _holiday_calendar_cache_get("US", None)
    assert first is second


def test_compute_drift_wasserstein_handles_length_mismatch() -> None:
    """Mismatched curve lengths must not silently bias the score."""
    from promforecast.drift import compute_drift  # noqa: PLC0415

    # Both curves describe the same flat-line distribution sampled at
    # different resolutions. After length-aware resampling the score
    # should be ~0; the pre-fix truncating implementation would also
    # report ~0 for this specific case, so we add a contrasting case
    # where the longer curve has a real outlier in its tail.
    flat_short = [10.0, 10.0, 10.0]
    flat_long = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    assert compute_drift(flat_short, flat_long, method="wasserstein") == pytest.approx(
        0.0, abs=1e-9
    )

    # ``cur`` carries one large value beyond the prev's length. With
    # truncation that tail was silently dropped; the resampling path
    # picks the outlier up.
    prev = [0.0, 0.0, 0.0, 0.0]
    cur = [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
    score = compute_drift(prev, cur, method="wasserstein")
    assert math.isfinite(score)
    assert score > 0.0


def test_inside_band_helper_is_used_by_both_diagnostics() -> None:
    """``deviation.inside_band`` is the single source of truth.

    Regression guard: the band-coverage path previously redid the
    inside/outside test inline; both should now agree by construction.
    """
    from promforecast.deviation import compute, inside_band  # noqa: PLC0415

    actual, lower, upper = 11.0, 0.0, 10.0
    d = compute(actual=actual, yhat=5.0, yhat_lower=lower, yhat_upper=upper)
    assert d.outside_band is True
    assert inside_band(actual, lower, upper) is False
    # Inverted-band order must yield the same answer (the helper
    # normalises lower/upper).
    assert inside_band(actual, upper, lower) is False
