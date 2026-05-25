"""Tests for the drift change-correlation explainer + runner wiring."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    DriftEmissionConfig,
    EmissionConfig,
    GroupConfig,
    QueryConfig,
    QueryEmissionConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.drift_explainer import (
    DriftWindow,
    classify_confidence,
    compute_pearson,
    explain,
)
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame

# ---- Pure-function tests ----------------------------------------------------


def test_pearson_perfect_positive_correlation() -> None:
    """Two identical signals correlate at +1.0."""
    coef, n = compute_pearson([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(coef, 1.0)
    assert n == 5


def test_pearson_perfect_negative_correlation() -> None:
    """Mirrored signals correlate at -1.0."""
    coef, n = compute_pearson([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0])
    assert math.isclose(coef, -1.0)
    assert n == 5


def test_pearson_drops_non_finite_pairs() -> None:
    """A single NaN doesn't poison the coefficient.

    Operators see NaN regressor readings when a fetch fails — we
    can't let that take down the correlation for the rest of the
    window.
    """
    coef, n = compute_pearson(
        [1.0, 2.0, float("nan"), 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert n == 4
    assert math.isclose(coef, 1.0)


def test_pearson_insufficient_samples_returns_nan() -> None:
    """Below the minimum pair count, the coefficient is NaN.

    The explainer treats this as ``insufficient_data`` and falls
    back to the value-based suspect.
    """
    coef, n = compute_pearson([1.0, 2.0], [1.0, 2.0])
    assert math.isnan(coef)
    assert n == 2


def test_pearson_zero_variance_returns_nan() -> None:
    """A flat regressor can't explain anything → NaN coefficient.

    Without this guard, a constant regressor would divide by zero
    inside the closed-form correlation; we surface NaN so the caller
    can fall back to the value suspect cleanly.
    """
    coef, n = compute_pearson([1.0, 2.0, 3.0, 4.0, 5.0], [7.0, 7.0, 7.0, 7.0, 7.0])
    assert math.isnan(coef)
    assert n == 5


def test_pearson_rejects_mismatched_lengths() -> None:
    """Programmer-error guard so a length bug fails loud."""
    with pytest.raises(ValueError):
        compute_pearson([1.0, 2.0], [1.0])


def test_classify_confidence_named_bands() -> None:
    """Confidence band matches the documented cutoffs."""
    assert classify_confidence(0.9, n_pairs=25) == "high"
    # Strong correlation but too few samples → drops to medium.
    assert classify_confidence(0.9, n_pairs=10) == "medium"
    # Medium correlation with enough samples → medium.
    assert classify_confidence(0.5, n_pairs=10) == "medium"
    # Weak correlation → low.
    assert classify_confidence(0.2, n_pairs=10) == "low"
    # Insufficient samples → distinct band, not "low".
    assert classify_confidence(0.9, n_pairs=3) == "insufficient_data"
    # NaN coefficient → insufficient_data.
    assert classify_confidence(float("nan"), n_pairs=100) == "insufficient_data"


def test_explain_picks_correlation_suspect_when_confidence_high() -> None:
    """High-confidence correlation wins over the loud-but-uncorrelated suspect.

    Captures the story 20 distinction: a regressor that's loud right
    now may be one-shot noise. The rolling-window correlation surfaces
    the genuine driver instead.
    """
    window = DriftWindow(maxlen=30)
    # Drift trends up over 20 samples — ``deploys`` regressor also
    # trends up (Pearson ≈ 1.0); ``noise`` regressor is random.
    for i in range(25):
        window.record(
            drift_score=i * 0.05,
            regressor_values={"deploys": float(i), "noise": float((i * 7) % 3)},
        )
    # Current values: "loud" is highest right now but doesn't track
    # the drift history.
    explanation = explain(
        window,
        current_regressor_values={"deploys": 24.0, "noise": 0.0, "loud": 9999.0},
    )
    assert explanation.top_correlation_suspect == "deploys"
    assert explanation.top_value_suspect == "loud"
    # Primary picks correlation when confidence is high.
    assert explanation.primary_suspect == "deploys"
    assert explanation.top_regressor_correlation_confidence == "high"


def test_explain_falls_back_to_value_suspect_when_insufficient_data() -> None:
    """A fresh-install drift event with no history uses the value suspect.

    The first time drift fires there's no rolling history to compute
    correlation against; primary_suspect picks the loud one so the
    operator gets *some* hypothesis to start with.
    """
    window = DriftWindow(maxlen=30)
    # No samples in the window — every correlation is undefined.
    explanation = explain(
        window,
        current_regressor_values={"deploys": 5.0, "noise": 1.0},
    )
    assert explanation.top_regressor_correlation_confidence == "insufficient_data"
    assert explanation.primary_suspect == "deploys"  # the loud one
    assert explanation.top_value_suspect == "deploys"


def test_explain_emits_unknown_when_no_regressors() -> None:
    """A query without regressors emits ``unknown`` rather than crashing."""
    window = DriftWindow(maxlen=30)
    explanation = explain(window, current_regressor_values={})
    assert explanation.primary_suspect == "unknown"
    assert explanation.top_value_suspect == "unknown"
    assert explanation.top_correlation_suspect == "unknown"


def test_drift_window_evicts_oldest_when_full() -> None:
    """The ring buffer respects ``maxlen``."""
    window = DriftWindow(maxlen=3)
    for i in range(5):
        window.record(drift_score=float(i), regressor_values={"r": float(i)})
    assert len(window) == 3
    samples = window.samples()
    assert [s.drift_score for s in samples] == [2.0, 3.0, 4.0]


# ---- Runner-level wiring ---------------------------------------------------


class _FakeSource(PromSource):
    def __init__(self) -> None:
        pass

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        return []

    async def aclose(self) -> None:
        return None


def _runner_config(alert_threshold: float = 0.25, correlation_window: int = 30) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["SeasonalNaive"],
            accuracy=AccuracyConfig(evaluate=False, auto_select=False),
            emission=EmissionConfig(quality=False, deviation=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="m",
                        promql="up",
                        emission=QueryEmissionConfig(
                            drift=DriftEmissionConfig(
                                enabled=True,
                                alert_threshold=alert_threshold,
                                correlation_window=correlation_window,
                            ),
                        ),
                    )
                ],
            )
        ],
    )


def _curve(model: str, query_id: str, values: list[float]) -> list[Any]:
    """Build minimal sink.ForecastCurvePoint shapes for the drift recorder."""
    from promforecast.sink import ForecastCurvePoint  # noqa: PLC0415

    points = []
    for i, v in enumerate(values):
        points.append(
            ForecastCurvePoint(
                metric=f"{query_id}_forecast",
                labels={"model": model, "instance": "host-a"},
                value=v,
                timestamp_ms=int((datetime(2026, 5, 21, tzinfo=UTC).timestamp() + i * 60) * 1000),
            )
        )
    return points


@pytest.mark.asyncio
async def test_record_drift_emits_explained_counter_and_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A drift crossing bumps the counter and emits the structured log event.

    Threads a synthetic curve through ``_record_drift`` twice — the
    first call seeds the cache; the second produces a drift score
    that crosses the configured threshold.
    """
    cfg = _runner_config(alert_threshold=0.1)
    runner = Runner(config=cfg, source=_FakeSource(), exporter=(exporter := Exporter()))
    from promforecast.point_factory import SeriesPointFactory  # noqa: PLC0415

    factory = SeriesPointFactory(
        metric_id="m",
        base_labels={"instance": "host-a"},
        group="g",
        levels=[],
    )
    from promforecast.runner._state import _FitResult  # noqa: PLC0415

    # Seed the cache with the first curve so the next call has a
    # baseline to compare against.
    seed = _FitResult()
    runner._record_drift(
        query=cfg.groups[0].queries[0],
        group_name="g",
        series_key="host-a",
        curve_points=_curve("m1", "m", [10.0, 10.0, 10.0]),
        factory=factory,
        result=seed,
        regressors=pd.DataFrame(
            {
                "unique_id": ["s"] * 3,
                "ds": pd.date_range("2026-01-01", periods=3, freq="5min"),
                "deploys": [0.0, 0.0, 1.0],
            }
        ),
    )
    # Second call: a curve that diverges sharply — score above the
    # 0.1 threshold so the explainer fires.
    second = _FitResult()
    runner._record_drift(
        query=cfg.groups[0].queries[0],
        group_name="g",
        series_key="host-a",
        curve_points=_curve("m1", "m", [100.0, 100.0, 100.0]),
        factory=factory,
        result=second,
        regressors=pd.DataFrame(
            {
                "unique_id": ["s"] * 3,
                "ds": pd.date_range("2026-01-01", periods=3, freq="5min"),
                "deploys": [0.0, 0.0, 5.0],
            }
        ),
    )
    assert second.drift_alerts == 1
    body = exporter.render().decode()
    assert "forecast_drift_explained_total{" in body
    assert 'group="g"' in body
    assert 'query="m"' in body
    # Primary suspect is "deploys" — the only configured regressor.
    assert 'primary_suspect="deploys"' in body
    # The structured log line ride structlog's configured renderer
    # (set up in main.run / tests indirectly); capsys catches the
    # stdout emission deterministically.
    captured = capsys.readouterr()
    assert "forecast_drift_explained" in (captured.out + captured.err)


@pytest.mark.asyncio
async def test_record_drift_with_no_regressors_emits_unknown_suspect() -> None:
    """A query without configured regressors still gets the counter row.

    The structured log line carries enough context that ``unknown``
    is itself a useful signal — "drift fired but I have no
    hypothesis for you". Without this, regressor-free deployments
    would never see the counter at all.
    """
    cfg = _runner_config(alert_threshold=0.1)
    runner = Runner(config=cfg, source=_FakeSource(), exporter=(exporter := Exporter()))
    from promforecast.point_factory import SeriesPointFactory  # noqa: PLC0415
    from promforecast.runner._state import _FitResult  # noqa: PLC0415

    factory = SeriesPointFactory(metric_id="m", base_labels={}, group="g", levels=[])
    runner._record_drift(
        query=cfg.groups[0].queries[0],
        group_name="g",
        series_key="host-a",
        curve_points=_curve("m1", "m", [10.0, 10.0]),
        factory=factory,
        result=_FitResult(),
        regressors=None,
    )
    runner._record_drift(
        query=cfg.groups[0].queries[0],
        group_name="g",
        series_key="host-a",
        curve_points=_curve("m1", "m", [50.0, 50.0]),
        factory=factory,
        result=_FitResult(),
        regressors=None,
    )
    body = exporter.render().decode()
    assert 'primary_suspect="unknown"' in body
